# ------------------------------------------------------------------------
# Modified from OFA (https://github.com/OFA-Sys/OFA)
# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.
# ------------------------------------------------------------------------
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import os.path
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import AdaptiveSoftmax, BaseLayer, FairseqDropout, LayerDropModuleList, LayerNorm, SinusoidalPositionalEmbedding, GradMultiply
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor

from .unify_transformer_layer import TransformerEncoderLayer, TransformerDecoderLayer
from .swin import SwinTransformer
from bert.modeling_bert import BertModel


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)


def BatchNorm2d(out_chan, momentum=0.1, eps=1e-3):
    return nn.SyncBatchNorm.convert_sync_batchnorm(nn.BatchNorm2d(out_chan, momentum=momentum, eps=eps))


def make_token_bucket_position(bucket_size, max_position=DEFAULT_MAX_SOURCE_POSITIONS):
    context_pos = torch.arange(max_position, dtype=torch.long)[:, None]
    memory_pos = torch.arange(max_position, dtype=torch.long)[None, :]
    relative_pos = context_pos - memory_pos
    sign = torch.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = torch.where((relative_pos < mid) & (relative_pos > -mid), mid - 1, torch.abs(relative_pos))
    log_pos = torch.ceil(torch.log(abs_pos / mid) / math.log((max_position - 1) / mid) * (mid - 1)) + mid
    log_pos = log_pos.int()
    bucket_pos = torch.where(abs_pos.le(mid), relative_pos, log_pos * sign).long()
    return bucket_pos + bucket_size - 1


def make_image_bucket_position(bucket_size, num_relative_distance):
    coords_h = torch.arange(bucket_size)
    coords_w = torch.arange(bucket_size)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += bucket_size - 1  # shift to start from 0
    relative_coords[:, :, 1] += bucket_size - 1
    relative_coords[:, :, 0] *= 2 * bucket_size - 1
    relative_position_index = torch.zeros(size=(bucket_size * bucket_size + 1,) * 2, dtype=relative_coords.dtype)
    relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    relative_position_index[0, 0:] = num_relative_distance - 3
    relative_position_index[0:, 0] = num_relative_distance - 2
    relative_position_index[0, 0] = num_relative_distance - 1
    return relative_position_index


@register_model("unify_transformer")
class TransformerModel(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
        self.supports_align_args = True
        self.use_depth = getattr(args, "use_depth", False)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                            help='decoder output dimension (extra linear layer '
                                 'if different from decoder embed dim')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        parser.add_argument('--checkpoint-activations', action='store_true',
                            help='checkpoint activations at each layer, which saves GPU '
                                 'memory usage at the cost of some additional compute')
        parser.add_argument('--offload-activations', action='store_true',
                            help='checkpoint activations at each layer, then save to gpu. Sets --checkpoint-activations.')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                            help='scalar quantization noise and scalar quantization at training time')
        # args for Fully Sharded Data Parallel (FSDP) training
        parser.add_argument(
            '--min-params-to-wrap', type=int, metavar='D', default=DEFAULT_MIN_PARAMS_TO_WRAP,
            help=(
                'minimum number of params for a layer to be wrapped with FSDP() when '
                'training with --ddp-backend=fully_sharded. Smaller values will '
                'improve memory efficiency, but may make torch.distributed '
                'communication less efficient due to smaller input sizes. This option '
                'is set to 0 (i.e., always wrap) when --checkpoint-activations or '
                '--offload-activations are passed.'
            )
        )

        parser.add_argument('--resnet-drop-path-rate', type=float,
                            help='resnet drop path rate')
        parser.add_argument('--encoder-drop-path-rate', type=float,
                            help='encoder drop path rate')
        parser.add_argument('--decoder-drop-path-rate', type=float,
                            help='encoder drop path rate')

        parser.add_argument('--token-bucket-size', type=int,
                            help='token bucket size')
        parser.add_argument('--image-bucket-size', type=int,
                            help='image bucket size')

        parser.add_argument('--attn-scale-factor', type=float,
                            help='attention scale factor')
        parser.add_argument('--freeze-resnet', action='store_true',
                            help='freeze resnet')
        parser.add_argument('--freeze-encoder-embedding', action='store_true',
                            help='freeze encoder token embedding')
        parser.add_argument('--freeze-decoder-embedding', action='store_true',
                            help='freeze decoder token embedding')
        parser.add_argument('--add-type-embedding', action='store_true',
                            help='add source/region/patch type embedding')

        parser.add_argument('--resnet-type', choices=['resnet50', 'resnet101', 'resnet152', 'swin-base'],
                            help='resnet type')
        parser.add_argument('--resnet-model-path', type=str, metavar='STR',
                            help='path to load resnet')
        parser.add_argument('--code-image-size', type=int,
                            help='code image size')
        parser.add_argument('--patch-layernorm-embedding', action='store_true',
                            help='add layernorm to patch embedding')
        parser.add_argument('--code-layernorm-embedding', action='store_true',
                            help='add layernorm to code embedding')
        parser.add_argument('--entangle-position-embedding', action='store_true',
                            help='entangle position embedding')
        parser.add_argument('--disable-entangle', action='store_true',
                            help='disable entangle')
        parser.add_argument('--sync-bn', action='store_true',
                            help='sync batchnorm')

        parser.add_argument('--scale-attn', action='store_true',
                            help='scale attn')
        parser.add_argument('--scale-fc', action='store_true',
                            help='scale fc')
        parser.add_argument('--scale-heads', action='store_true',
                            help='scale heads')
        parser.add_argument('--scale-resids', action='store_true',
                            help='scale resids')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError("--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim")
            if args.decoder_embed_path and (args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError("--share-all-embeddings not compatible with --decoder-embed-path")
            encoder_embed_tokens = cls.build_embedding(args, src_dict, args.encoder_embed_dim, args.encoder_embed_path)
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(args, src_dict, args.encoder_embed_dim, args.encoder_embed_path)
            decoder_embed_tokens = cls.build_embedding(args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path)
        if getattr(args, "freeze_encoder_embedding", False):
            encoder_embed_tokens.weight.requires_grad = False
        if getattr(args, "freeze_decoder_embedding", False):
            decoder_embed_tokens.weight.requires_grad = False
        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        if not args.share_all_embeddings:
            min_params_to_wrap = getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            # fsdp_wrap is a no-op when --ddp-backend != fully_sharded
            encoder = fsdp_wrap(encoder, min_num_params=min_params_to_wrap)
            decoder = fsdp_wrap(decoder, min_num_params=min_params_to_wrap)
        return cls(args, encoder, decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        att_masks,
        prev_output_tokens_11,
        prev_output_tokens_12,
        prev_output_tokens_21,
        prev_output_tokens_22,
        delta_x1,
        delta_y1,
        delta_x2,
        delta_y2,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens)
        decoder_out = self.decoder(
            prev_output_tokens_11,
            prev_output_tokens_12,
            prev_output_tokens_21,
            prev_output_tokens_22,
            delta_x1,
            delta_y1,
            delta_x2,
            delta_y2,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.encoder_layerdrop = args.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions
        self.num_attention_heads = args.encoder_attention_heads

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)
        
        self.use_depth = args.use_depth

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        if getattr(args, "add_type_embedding", False):
            ### consider depth input
            if self.use_depth:
                self.type_embedding = Embedding(3, embed_dim, padding_idx=None)
            else:
                self.type_embedding = Embedding(2, embed_dim, padding_idx=None)
        else:
            self.type_embedding = None

        conv_dim = 1024
        if args.vis_encoder_type == "swin-base":
            out_index = args.out_index
            self.embed_images = SwinTransformer(
                pretrain_img_size=384, window_size=12, embed_dim=128, out_indices=[out_index], depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32]
            )
            if out_index == 2:
                conv_dim = 512
            ckpt_path = "../../pretrained_weights/swin_base_patch4_window12_384_22k.pth"
            if os.path.exists(ckpt_path):
                self.embed_images.init_weights(pretrained=ckpt_path)
                print("Loaded Swin Pretrained Weights", ckpt_path)
        elif args.vis_encoder_type == "swin-large":
            out_indices = args.out_index
            self.embed_images = SwinTransformer(
                pretrain_img_size=384, window_size=12, embed_dim=192, out_indices=[out_indices], depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48]
            )
            conv_dim = 768 if out_indices == 2 else 1536
            ckpt_path = "../../pretrained_weights/swin_large_patch4_window12_384_22k.pth"
            if os.path.exists(ckpt_path):
                self.embed_images.init_weights(pretrained=ckpt_path)
                print("Loaded Swin Pretrained Weights", ckpt_path)
        else:
            raise NotImplementedError

        # depth encoder
        if self.use_depth:
            from .convnext import ConvNeXt4Depth

            self.depth_encoder = ConvNeXt4Depth.build_tiny()
        # end of depth encoder

        self.image_proj = Linear(conv_dim, embed_dim)
        if getattr(args, "patch_layernorm_embedding", False):
            self.patch_layernorm_embedding = LayerNorm(embed_dim)
            if self.use_depth:
                self.depth_layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.patch_layernorm_embedding = None

        self.embed_positions = Embedding(args.max_source_positions + 2, embed_dim)
        self.embed_image_positions = Embedding(args.image_bucket_size**2 + 1, embed_dim)
        self.pos_ln = LayerNorm(embed_dim)
        self.image_pos_ln = LayerNorm(embed_dim)
        self.pos_scaling = float(embed_dim / args.encoder_attention_heads * args.attn_scale_factor) ** -0.5
        self.pos_q_linear = nn.Linear(embed_dim, embed_dim)
        self.pos_k_linear = nn.Linear(embed_dim, embed_dim)

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])

        dpr = [x.item() for x in torch.linspace(0, args.encoder_drop_path_rate, args.encoder_layers)]
        self.layers.extend([self.build_encoder_layer(args, drop_path_rate=dpr[i]) for i in range(args.encoder_layers)])
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        token_bucket_size = args.token_bucket_size
        token_num_rel_dis = 2 * token_bucket_size - 1
        token_rp_bucket = make_token_bucket_position(token_bucket_size)
        self.token_rel_pos_table_list = nn.ModuleList(
            [Embedding(token_num_rel_dis, self.num_attention_heads, zero_init=True) for _ in range(args.encoder_layers)]
        )

        image_bucket_size = args.image_bucket_size
        image_num_rel_dis = (2 * image_bucket_size - 1) * (2 * image_bucket_size - 1) + 3
        image_rp_bucket = make_image_bucket_position(image_bucket_size, image_num_rel_dis)
        self.image_rel_pos_table_list = nn.ModuleList(
            [Embedding(image_num_rel_dis, self.num_attention_heads, zero_init=True) for _ in range(args.encoder_layers)]
        )

        self.register_buffer("token_rp_bucket", token_rp_bucket)
        self.register_buffer("image_rp_bucket", image_rp_bucket)
        self.entangle_position_embedding = args.entangle_position_embedding
        # self.bert = BertModel.from_pretrained("bert-base-uncased")
        from bert.configuration_bert import BertConfig

        self.bert = BertModel(BertConfig())

    def train(self, mode=True):
        super(TransformerEncoder, self).train(mode)
        if getattr(self.args, "freeze_resnet", False):
            for m in self.embed_images.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def build_encoder_layer(self, args, drop_path_rate=0.0):
        layer = TransformerEncoderLayer(args, drop_path_rate=drop_path_rate)
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP) if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def get_rel_pos_bias(self, x, idx):
        seq_len = x.size(1)
        rp_bucket = self.token_rp_bucket[:seq_len, :seq_len]
        values = F.embedding(rp_bucket, self.token_rel_pos_table_list[idx].weight)
        values = values.unsqueeze(0).expand(x.size(0), -1, -1, -1)
        values = values.permute([0, 3, 1, 2])
        return values.contiguous()

    def get_image_rel_pos_bias(self, image_position_ids, idx):
        bsz, seq_len = image_position_ids.shape
        rp_bucket_size = self.image_rp_bucket.size(1)

        rp_bucket = (
            self.image_rp_bucket.unsqueeze(0)
            .expand(bsz, rp_bucket_size, rp_bucket_size)
            .gather(1, image_position_ids[:, :, None].expand(bsz, seq_len, rp_bucket_size))
            .gather(2, image_position_ids[:, None, :].expand(bsz, seq_len, seq_len))
        )
        values = F.embedding(rp_bucket, self.image_rel_pos_table_list[idx].weight)
        values = values.permute(0, 3, 1, 2)
        return values

    def get_patch_images_info(self, patch_images, sample_patch_num, device):
        image_embed = self.embed_images(patch_images)
        h, w = image_embed.shape[-2:]  # (B, C, H, W) (B, 1024, 16, 16)
        image_num_patches = h * w
        image_padding_mask = patch_images.new_zeros((patch_images.size(0), image_num_patches)).bool()
        image_position_idx = torch.arange(w).unsqueeze(0).expand(h, w) + torch.arange(h).unsqueeze(1) * self.args.image_bucket_size + 1
        image_position_idx = image_position_idx.view(-1).to(device)
        image_position_ids = image_position_idx[None, :].expand(patch_images.size(0), image_num_patches)

        image_embed = image_embed.flatten(2).transpose(1, 2)
        if sample_patch_num is not None:
            patch_orders = [random.sample(range(image_num_patches), k=sample_patch_num) for _ in range(patch_images.size(0))]
            patch_orders = torch.LongTensor(patch_orders).to(device)
            image_embed = image_embed.gather(1, patch_orders.unsqueeze(2).expand(-1, -1, image_embed.size(2)))
            image_num_patches = sample_patch_num
            image_padding_mask = image_padding_mask.gather(1, patch_orders)
            image_position_ids = image_position_ids.gather(1, patch_orders)
        image_pos_embed = self.embed_image_positions(image_position_ids)

        return image_embed, image_num_patches, image_padding_mask, image_position_ids, image_pos_embed

    def forward_embedding(
        self,
        src_tokens,
        att_masks,
        image_embed: Optional[torch.Tensor] = None,
        token_embedding: Optional[torch.Tensor] = None,
        pos_embed: Optional[torch.Tensor] = None,
        image_pos_embed: Optional[torch.Tensor] = None,
        depth_features: Optional[torch.Tensor] = None,
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.bert(src_tokens, attention_mask=att_masks)[0]

        x = embed = token_embedding
        if self.entangle_position_embedding and pos_embed is not None:
            x += pos_embed
        if self.type_embedding is not None:
            x += self.type_embedding(src_tokens.new_zeros(x.size()[:2]))
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)

        # embed raw images
        if image_embed is not None:
            image_embed = self.image_proj(image_embed)
            image_x = image_embed = image_embed
            if self.entangle_position_embedding and image_pos_embed is not None:
                image_x += image_pos_embed
            if self.type_embedding is not None:
                image_x += self.type_embedding(src_tokens.new_ones(image_x.size()[:2]))
            if self.patch_layernorm_embedding is not None:
                image_x = self.patch_layernorm_embedding(image_x)
            image_x = self.dropout_module(image_x)
            if self.quant_noise is not None:
                image_x = self.quant_noise(image_x)

            # embed depth features
            if self.use_depth:
                if self.entangle_position_embedding and image_pos_embed is not None:
                    depth_features += image_pos_embed
                if self.type_embedding is not None:
                    # type is 2
                    depth_features += self.type_embedding(src_tokens.new_full(depth_features.size()[:2], 2))
                if self.patch_layernorm_embedding is not None:
                    depth_features = self.depth_layernorm_embedding(depth_features)
                x = torch.cat([depth_features, x], dim=1)
                embed = torch.cat([depth_features, embed], dim=1)

            x = torch.cat([image_x, x], dim=1)
            embed = torch.cat([image_embed, embed], dim=1)

        return x, embed

    def forward(
        self,
        src_tokens,
        src_lengths,
        att_masks,
        patch_images: Optional[torch.Tensor] = None,
        patch_depths: Optional[torch.Tensor] = None,
        patch_masks: Optional[torch.Tensor] = None,
        code_masks: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        sample_patch_num: Optional[int] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        return self.forward_scriptable(
            src_tokens, src_lengths, att_masks, patch_images, patch_depths, patch_masks, return_all_hiddens, token_embeddings, sample_patch_num
        )

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
        self,
        src_tokens,
        src_lengths,
        att_masks,
        patch_images: Optional[torch.Tensor] = None,
        patch_depths: Optional[torch.Tensor] = None,
        patch_masks: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        sample_patch_num: Optional[int] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        ### check use_depth
        if self.use_depth and patch_depths is None:
            raise ValueError("depth_images is required when use_depth is True")
        ### end of check use_depth

        ### depth encoder
        if self.use_depth:
            depth_features = self.depth_encoder(patch_depths)  # (B, C, H, W) (B, 1024, 16, 16)
            # B, C, H, W -> B, H*W, C
            depth_features = depth_features.flatten(2).transpose(1, 2)
        else:
            depth_features = None
        ### end of depth encoder

        image_embed = None
        image_pos_embed = None
        if patch_images is not None:
            image_embed, image_num_patches, image_padding_mask, image_position_ids, image_pos_embed = self.get_patch_images_info(
                patch_images, sample_patch_num, src_tokens.device
            )
            image_padding_mask[~patch_masks] = True

        encoder_padding_mask = src_tokens.eq(0)
        # encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if patch_images is not None:
            if self.use_depth:
                encoder_padding_mask = torch.cat([image_padding_mask, encoder_padding_mask], dim=1)
            encoder_padding_mask = torch.cat([image_padding_mask, encoder_padding_mask], dim=1)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        pos_embed = self.embed_positions(utils.new_arange(src_tokens))
        x, encoder_embedding = self.forward_embedding(
            src_tokens, att_masks, image_embed, token_embeddings, pos_embed, image_pos_embed, depth_features=depth_features
        )

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        pos_embed = self.pos_ln(pos_embed)
        if patch_images is not None:
            image_pos_embed = self.image_pos_ln(image_pos_embed)
            pos_embed = torch.cat([image_pos_embed, pos_embed], dim=1)
            if self.use_depth:
                pos_embed = torch.cat([image_pos_embed, pos_embed], dim=1)

        pos_q = self.pos_q_linear(pos_embed).view(x.size(1), x.size(0), self.num_attention_heads, -1).transpose(1, 2) * self.pos_scaling
        pos_k = self.pos_k_linear(pos_embed).view(x.size(1), x.size(0), self.num_attention_heads, -1).transpose(1, 2)
        abs_pos_bias = torch.matmul(pos_q, pos_k.transpose(2, 3))

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for idx, layer in enumerate(self.layers):
            self_attn_bias = abs_pos_bias.clone()
            self_attn_bias[:, :, -src_tokens.size(1) :, -src_tokens.size(1) :] += self.get_rel_pos_bias(src_tokens, idx)

            if patch_images is not None:
                if self.use_depth:
                    # [rgb, depth, text] rgb and depth have same dimension
                    len_rgb_tokens = (x.size(0) - src_tokens.size(1)) // 2
                    len_depth_tokens = len_rgb_tokens
                    self_attn_bias[:, :, :len_rgb_tokens, :len_rgb_tokens] += self.get_image_rel_pos_bias(image_position_ids, idx)
                    self_attn_bias[
                        :, :, len_rgb_tokens : len_rgb_tokens + len_depth_tokens, len_rgb_tokens : len_rgb_tokens + len_depth_tokens
                    ] += self.get_image_rel_pos_bias(image_position_ids, idx)
                else:
                    # [rgb, text]
                    self_attn_bias[:, :, : x.size(0) - src_tokens.size(1), : x.size(0) - src_tokens.size(1)] += self.get_image_rel_pos_bias(
                        image_position_ids, idx
                    )
            self_attn_bias = self_attn_bias.reshape(-1, x.size(0), x.size(0))

            x = layer(x, encoder_padding_mask=encoder_padding_mask if has_pads else None, self_attn_bias=self_attn_bias)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
            "position_embeddings": [pos_embed],  # B x T x C
        }

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [encoder_out["encoder_padding_mask"][0].index_select(0, new_order)]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [encoder_out["encoder_embedding"][0].index_select(0, new_order)]

        if len(encoder_out["src_tokens"]) == 0:
            new_src_tokens = []
        else:
            new_src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            new_src_lengths = []
        else:
            new_src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        if len(encoder_out["position_embeddings"]) == 0:
            new_position_embeddings = []
        else:
            new_position_embeddings = [(encoder_out["position_embeddings"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": new_src_tokens,  # B x T
            "src_lengths": new_src_lengths,  # B x 1
            "position_embeddings": new_position_embeddings,  # B x T x C
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return self.max_source_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict["{}.embed_positions._float_tensor".format(name)] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(state_dict, "{}.layers.{}".format(name, i))

        # version_key = "{}.version".format(name)
        # if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
        #     # earlier checkpoints did not normalize after the stack of layers
        #     self.layer_norm = None
        #     self.normalize = False
        #     state_dict[version_key] = torch.Tensor([1])

        prefix = name + "." if name != "" else ""
        for param_name, param_tensor in self.state_dict().items():
            if (prefix + param_name) not in state_dict:
                state_dict[prefix + param_name] = self.state_dict()[param_name]

        if len(state_dict["encoder.embed_image_positions.weight"]) < len(self.state_dict()["embed_image_positions.weight"]):
            num_posids_to_add = len(self.state_dict()["embed_image_positions.weight"]) - len(state_dict["encoder.embed_image_positions.weight"])
            embed_dim = state_dict["encoder.embed_image_positions.weight"].size(1)
            new_pos_embed_to_add = torch.zeros(num_posids_to_add, embed_dim)
            nn.init.normal_(new_pos_embed_to_add, mean=0, std=embed_dim**-0.5)
            new_pos_embed_to_add = new_pos_embed_to_add.to(
                dtype=state_dict["encoder.embed_image_positions.weight"].dtype,
            )
            state_dict["encoder.embed_image_positions.weight"] = torch.cat([state_dict["encoder.embed_image_positions.weight"], new_pos_embed_to_add])
        return state_dict


class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed
        self.num_attention_heads = args.decoder_attention_heads

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.window_size = args.code_image_size // 8

        self.embed_positions = Embedding(args.max_target_positions + 2, embed_dim)
        self.embed_image_positions = Embedding(args.image_bucket_size**2 + 1, embed_dim)
        self.pos_ln = LayerNorm(embed_dim)
        self.image_pos_ln = LayerNorm(embed_dim)
        self.pos_scaling = float(embed_dim / self.num_attention_heads * args.attn_scale_factor) ** -0.5
        self.self_pos_q_linear = nn.Linear(embed_dim, embed_dim)
        self.self_pos_k_linear = nn.Linear(embed_dim, embed_dim)
        self.cross_pos_q_linear = nn.Linear(embed_dim, embed_dim)
        self.cross_pos_k_linear = nn.Linear(embed_dim, embed_dim)

        if getattr(args, "code_layernorm_embedding", False):
            self.code_layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.code_layernorm_embedding = None

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])

        dpr = [x.item() for x in torch.linspace(0, args.decoder_drop_path_rate, args.decoder_layers)]
        self.layers.extend([self.build_decoder_layer(args, no_encoder_attn, drop_path_rate=dpr[i]) for i in range(args.decoder_layers)])
        self.num_layers = len(self.layers)

        if args.decoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False) if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights else None
        )

        self.adaptive_softmax = None
        self.reg_head = output_projection
        if self.reg_head is None:
            self.build_output_projection(args, dictionary, embed_tokens)

        token_bucket_size = args.token_bucket_size
        token_num_rel_dis = 2 * token_bucket_size - 1
        token_rp_bucket = make_token_bucket_position(token_bucket_size)
        self.token_rel_pos_table_list = nn.ModuleList(
            [Embedding(token_num_rel_dis, self.num_attention_heads, zero_init=True) for _ in range(args.decoder_layers)]
        )

        image_bucket_size = args.image_bucket_size
        image_num_rel_dis = (2 * image_bucket_size - 1) * (2 * image_bucket_size - 1) + 3
        image_rp_bucket = make_image_bucket_position(image_bucket_size, image_num_rel_dis)
        image_position_idx = (
            torch.arange(self.window_size).unsqueeze(0).expand(self.window_size, self.window_size)
            + torch.arange(self.window_size).unsqueeze(1) * image_bucket_size
            + 1
        )
        image_position_idx = torch.cat([torch.tensor([0]), image_position_idx.view(-1)])
        image_position_idx = torch.cat([image_position_idx, torch.tensor([1024] * 768)])
        self.image_rel_pos_table_list = nn.ModuleList(
            [Embedding(image_num_rel_dis, self.num_attention_heads, zero_init=True) for _ in range(args.decoder_layers)]
        )

        self.register_buffer("token_rp_bucket", token_rp_bucket)
        self.register_buffer("image_rp_bucket", image_rp_bucket)
        self.register_buffer("image_position_idx", image_position_idx)
        self.entangle_position_embedding = args.entangle_position_embedding

    def build_output_projection(self, args, dictionary, embed_tokens):
        self.reg_head = MLP(self.output_embed_dim, self.output_embed_dim, 2, 3)
        nn.init.constant_(self.reg_head.layers[-1].weight.data, 0)
        nn.init.constant_(self.reg_head.layers[-1].bias.data, 0)

        # classify token types
        self.cls_head = nn.Linear(self.output_embed_dim, 3, bias=False)  # 3 types: coordinate, polygon separator, eos
        nn.init.normal_(self.cls_head.weight, mean=0, std=self.output_embed_dim**-0.5)

        num_base_layers = getattr(args, "base_layers", 0)
        for i in range(num_base_layers):
            self.layers.insert(((i + 1) * args.decoder_layers) // (num_base_layers + 1), BaseLayer(args))

    def build_decoder_layer(self, args, no_encoder_attn=False, drop_path_rate=0.0):
        layer = TransformerDecoderLayer(args, no_encoder_attn, drop_path_rate=drop_path_rate)
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP) if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def get_rel_pos_bias(self, x, idx):
        seq_len = x.size(1)
        rp_bucket = self.token_rp_bucket[:seq_len, :seq_len]
        values = F.embedding(rp_bucket, self.token_rel_pos_table_list[idx].weight)
        values = values.permute([2, 0, 1])
        return values.contiguous()

    def get_image_rel_pos_bias(self, x, idx):
        seq_len = x.size(1)
        image_position_idx = self.image_position_idx[:seq_len]
        rp_bucket = self.image_rp_bucket[image_position_idx][:, image_position_idx]
        values = F.embedding(rp_bucket, self.image_rel_pos_table_list[idx].weight)
        values = values.permute(2, 0, 1)
        return values

    def get_pos_info(self, tokens, tgt_pos_embed, src_pos_embed=None, use_image=False):
        batch_size = tokens.size(0)
        tgt_len = tokens.size(1)
        tgt_pos_embed = self.image_pos_ln(tgt_pos_embed) if use_image else self.pos_ln(tgt_pos_embed)
        if src_pos_embed is not None:
            src_len = src_pos_embed.size(1)
            pos_q = self.cross_pos_q_linear(tgt_pos_embed).view(batch_size, tgt_len, self.num_attention_heads, -1).transpose(1, 2) * self.pos_scaling
            pos_k = self.cross_pos_k_linear(src_pos_embed).view(batch_size, src_len, self.num_attention_heads, -1).transpose(1, 2)
        else:
            src_len = tgt_pos_embed.size(1)
            pos_q = self.self_pos_q_linear(tgt_pos_embed).view(batch_size, tgt_len, self.num_attention_heads, -1).transpose(1, 2) * self.pos_scaling
            pos_k = self.self_pos_k_linear(tgt_pos_embed).view(batch_size, src_len, self.num_attention_heads, -1).transpose(1, 2)
        abs_pos_bias = torch.matmul(pos_q, pos_k.transpose(2, 3))
        return abs_pos_bias

    def forward(
        self,
        prev_output_tokens_11,
        prev_output_tokens_12,
        prev_output_tokens_21,
        prev_output_tokens_22,
        delta_x1,
        delta_y1,
        delta_x2,
        delta_y2,
        code_masks: Optional[torch.Tensor] = None,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        x, extra = self.extract_features(
            prev_output_tokens_11,
            prev_output_tokens_12,
            prev_output_tokens_21,
            prev_output_tokens_22,
            delta_x1,
            delta_y1,
            delta_x2,
            delta_y2,
            code_masks=code_masks,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        x1 = x
        x2 = None
        if not features_only:
            x1, x2 = self.output_layer(x)
        return x1, x2, extra

    def extract_features(
        self,
        prev_output_tokens_11,
        prev_output_tokens_12,
        prev_output_tokens_21,
        prev_output_tokens_22,
        delta_x1,
        delta_y1,
        delta_x2,
        delta_y2,
        code_masks: Optional[torch.Tensor],
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens_11,
            prev_output_tokens_12,
            prev_output_tokens_21,
            prev_output_tokens_22,
            delta_x1,
            delta_y1,
            delta_x2,
            delta_y2,
            code_masks,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. A copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens_11,
        prev_output_tokens_12,
        prev_output_tokens_21,
        prev_output_tokens_22,
        delta_x1,
        delta_y1,
        delta_x2,
        delta_y2,
        code_masks: Optional[torch.Tensor],
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """

        prev_output_tokens = prev_output_tokens_11

        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert enc.size()[1] == bs, f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        bsz, tgt_len = prev_output_tokens.shape
        token_position_idx = utils.new_arange(prev_output_tokens)
        tgt_pos_embed = self.embed_positions(token_position_idx)
        if code_masks is not None and torch.any(code_masks):
            image_position_idx = self.image_position_idx[: prev_output_tokens.size(1)].unsqueeze(0).expand(bsz, tgt_len)
            tgt_pos_embed[code_masks] = self.embed_image_positions(image_position_idx)[code_masks]

        # self attn position bias
        self_abs_pos_bias = self.get_pos_info(prev_output_tokens, tgt_pos_embed, use_image=False)
        if code_masks is not None and torch.any(code_masks):
            self_image_abs_pos_bias = self.get_pos_info(prev_output_tokens, tgt_pos_embed, use_image=True)
            self_abs_pos_bias[code_masks] = self_image_abs_pos_bias[code_masks]
        # cross attn position bias
        src_pos_embed = encoder_out["position_embeddings"][0]
        cross_abs_pos_bias = self.get_pos_info(prev_output_tokens, tgt_pos_embed, src_pos_embed=src_pos_embed)
        if code_masks is not None and torch.any(code_masks):
            cross_image_abs_pos_bias = self.get_pos_info(prev_output_tokens, tgt_pos_embed, src_pos_embed=src_pos_embed, use_image=True)
            cross_abs_pos_bias[code_masks] = cross_image_abs_pos_bias[code_masks]
        cross_abs_pos_bias = cross_abs_pos_bias.reshape(-1, *cross_abs_pos_bias.size()[-2:])

        all_prev_output_tokens = prev_output_tokens.clone()
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            cross_abs_pos_bias = cross_abs_pos_bias[:, -1:, :]
            tgt_pos_embed = tgt_pos_embed[:, -1:, :]

        # embed tokens and positions
        token_embedding_11 = self.embed_tokens(prev_output_tokens_11)
        token_embedding_12 = self.embed_tokens(prev_output_tokens_12)
        token_embedding_21 = self.embed_tokens(prev_output_tokens_21)
        token_embedding_22 = self.embed_tokens(prev_output_tokens_22)
        delta_x1 = delta_x1.unsqueeze(-1).repeat(1, 1, token_embedding_11.shape[-1])
        delta_x2 = delta_x2.unsqueeze(-1).repeat(1, 1, token_embedding_11.shape[-1])
        delta_y1 = delta_y1.unsqueeze(-1).repeat(1, 1, token_embedding_11.shape[-1])
        delta_y2 = delta_y2.unsqueeze(-1).repeat(1, 1, token_embedding_11.shape[-1])

        token_embedding = (
            token_embedding_11 * delta_x2 * delta_y2
            + token_embedding_12 * delta_x2 * delta_y1
            + token_embedding_21 * delta_x1 * delta_y2
            + token_embedding_22 * delta_x1 * delta_y1
        )

        x = self.embed_scale * token_embedding

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if self.entangle_position_embedding is not None and not self.args.disable_entangle:
            x += tgt_pos_embed

        if self.layernorm_embedding is not None:
            if code_masks is None or not code_masks.any() or not getattr(self, "code_layernorm_embedding", False):
                x = self.layernorm_embedding(x.half())
            elif code_masks is not None and code_masks.all():
                x = self.code_layernorm_embedding(x)
            else:
                x[~code_masks] = self.layernorm_embedding(x[~code_masks])
                x[code_masks] = self.code_layernorm_embedding(x[code_masks])

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            self_attn_bias = self_abs_pos_bias.clone()
            if code_masks is None or not code_masks.any():
                self_attn_bias += self.get_rel_pos_bias(all_prev_output_tokens, idx).unsqueeze(0)
            elif code_masks is not None and code_masks.all():
                self_attn_bias += self.get_image_rel_pos_bias(all_prev_output_tokens, idx).unsqueeze(0)
            else:
                self_attn_bias[~code_masks] += self.get_rel_pos_bias(all_prev_output_tokens, idx).unsqueeze(0)
                self_attn_bias[code_masks] += self.get_image_rel_pos_bias(all_prev_output_tokens, idx).unsqueeze(0)
            self_attn_bias = self_attn_bias.reshape(-1, *self_attn_bias.size()[-2:])
            if incremental_state is not None:
                self_attn_bias = self_attn_bias[:, -1:, :]

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
                self_attn_bias=self_attn_bias,
                cross_attn_bias=cross_abs_pos_bias,
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.cls_head(features), F.sigmoid(self.reg_head(features))
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return self.max_target_positions

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if self._future_mask.size(0) == 0 or (not self._future_mask.device == tensor.device) or self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1)
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict["{}.embed_positions._float_tensor".format(name)] = torch.FloatTensor(1)

        if f"{name}.output_projection.weight" not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f"{name}.embed_tokens.weight"
            else:
                embed_out_key = f"{name}.embed_out"
            if embed_out_key in state_dict:
                state_dict[f"{name}.output_projection.weight"] = state_dict[embed_out_key]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(state_dict, "{}.layers.{}".format(name, i))

        prefix = name + "." if name != "" else ""
        image_params = ["image_position_idx"]
        for image_param in image_params:
            state_dict[prefix + image_param] = self.state_dict()[image_param]
        for param_name, param_tensor in self.state_dict().items():
            if (prefix + param_name) not in state_dict:
                state_dict[prefix + param_name] = self.state_dict()[param_name]

        if len(state_dict["decoder.embed_image_positions.weight"]) < len(self.state_dict()["embed_image_positions.weight"]):
            num_posids_to_add = len(self.state_dict()["embed_image_positions.weight"]) - len(state_dict["decoder.embed_image_positions.weight"])
            embed_dim = state_dict["decoder.embed_image_positions.weight"].size(1)
            new_pos_embed_to_add = torch.zeros(num_posids_to_add, embed_dim)
            nn.init.normal_(new_pos_embed_to_add, mean=0, std=embed_dim**-0.5)
            new_pos_embed_to_add = new_pos_embed_to_add.to(
                dtype=state_dict["decoder.embed_image_positions.weight"].dtype,
            )
            state_dict["decoder.embed_image_positions.weight"] = torch.cat([state_dict["decoder.embed_image_positions.weight"], new_pos_embed_to_add])
        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx=None, zero_init=False):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    if zero_init:
        nn.init.constant_(m.weight, 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


@register_model_architecture("unify_transformer", "unify_transformer")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(args, "share_decoder_input_output_embed", False)
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(args, "no_token_positional_embeddings", False)
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(args, "decoder_output_dim", args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
