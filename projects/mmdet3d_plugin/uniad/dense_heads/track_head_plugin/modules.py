import torch
import torch.nn.functional as F
from torch import nn
from .track_instance import Instances

def index_bool2long_trt(bool_index):
    # Convert Boolean Tensor to Long Tensor:
    long_index = bool_index.long()
    # Create a Range Tensor and Multiply by the Long Tensor
    long_index = torch.arange(1, long_index.shape[-1]+1
                                        , device=long_index.device)*long_index
    # Extract Non-Zero Indices
    long_index = long_index[long_index.nonzero(as_tuple=True)[0]]
    # Adjust Indices to Be Zero-Based
    long_index = long_index - torch.ones_like(long_index)
    return long_index

# MemoryBank
class MemoryBank(nn.Module):

    def __init__(self,
                 args,
                 dim_in, hidden_dim, dim_out,
                 ):
        super().__init__()
        self._build_layers(args, dim_in, hidden_dim, dim_out)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        self.save_thresh = args['memory_bank_score_thresh']
        self.save_period = 3
        self.max_his_length = args['memory_bank_len']

        self.save_proj = nn.Linear(dim_in, dim_in)

        self.temporal_attn = nn.MultiheadAttention(dim_in, 8, dropout=0)
        self.temporal_fc1 = nn.Linear(dim_in, hidden_dim)
        self.temporal_fc2 = nn.Linear(hidden_dim, dim_in)
        self.temporal_norm1 = nn.LayerNorm(dim_in)
        self.temporal_norm2 = nn.LayerNorm(dim_in)

    def update(self, track_instances):
        embed = track_instances.output_embedding[:, None]  #( N, 1, 256)
        scores = track_instances.scores
        mem_padding_mask = track_instances.mem_padding_mask
        device = embed.device

        save_period = track_instances.save_period
        if self.training:
            saved_idxes = scores > 0
        else:
            saved_idxes = (save_period == 0) & (scores > self.save_thresh)
            # saved_idxes = (save_period == 0)
            save_period[save_period > 0] -= 1
            save_period[saved_idxes] = self.save_period

        saved_embed = embed[saved_idxes]
        if len(saved_embed) > 0:
            prev_embed = track_instances.mem_bank[saved_idxes]
            save_embed = self.save_proj(saved_embed)
            mem_padding_mask[saved_idxes] = torch.cat([mem_padding_mask[saved_idxes, 1:], torch.zeros((len(saved_embed), 1), dtype=torch.bool, device=device)], dim=1)
            track_instances.mem_bank = track_instances.mem_bank.clone()
            track_instances.mem_bank[saved_idxes] = torch.cat([prev_embed[:, 1:], save_embed], dim=1)

    def _forward_temporal_attn(self, track_instances):
        if len(track_instances) == 0:
            return track_instances

        key_padding_mask = track_instances.mem_padding_mask  # [n_, memory_bank_len]

        valid_idxes = key_padding_mask[:, -1] == 0
        embed = track_instances.output_embedding[valid_idxes]  # (n, 256)

        if len(embed) > 0:
            prev_embed = track_instances.mem_bank[valid_idxes]
            key_padding_mask = key_padding_mask[valid_idxes]
            embed2 = self.temporal_attn(
                embed[None],                  # (num_track, dim) to (1, num_track, dim)
                prev_embed.transpose(0, 1),   # (num_track, mem_len, dim) to (mem_len, num_track, dim)
                prev_embed.transpose(0, 1),
                key_padding_mask=key_padding_mask,
            )[0][0]

            embed = self.temporal_norm1(embed + embed2)
            embed2 = self.temporal_fc2(F.relu(self.temporal_fc1(embed)))
            embed = self.temporal_norm2(embed + embed2)
            track_instances.output_embedding = track_instances.output_embedding.clone()
            track_instances.output_embedding[valid_idxes] = embed

        return track_instances

    def forward_temporal_attn(self, track_instances):
        return self._forward_temporal_attn(track_instances)

    def forward(self, track_instances: Instances, update_bank=True) -> Instances:
        track_instances = self._forward_temporal_attn(track_instances)
        if update_bank:
            self.update(track_instances)
        return track_instances


# MemoryBankTRTP
class MemoryBankTRTP(MemoryBank):

    def __init__(self,
                 args,
                 dim_in, hidden_dim, dim_out,
                 ):
        super(MemoryBank, self).__init__()
        self._build_layers(args, dim_in, hidden_dim, dim_out)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        self.save_thresh = args['memory_bank_score_thresh']
        self.save_period = 3
        self.max_his_length = args['memory_bank_len']

        self.save_proj = nn.Linear(dim_in, dim_in)

        self.temporal_attn = nn.MultiheadAttention(dim_in, 8, dropout=0)
        self.temporal_fc1 = nn.Linear(dim_in, hidden_dim)
        self.temporal_fc2 = nn.Linear(hidden_dim, dim_in)
        self.temporal_norm1 = nn.LayerNorm(dim_in)
        self.temporal_norm2 = nn.LayerNorm(dim_in)

    def update_trt(self, output_embedding, scores, mem_padding_mask, save_period, mem_bank):
        embed = output_embedding[:, None]  #( N, 1, 256)
        saved_idxes = index_bool2long_trt((save_period == 0) & (scores > self.save_thresh))
        save_period[index_bool2long_trt(save_period > 0)] = \
            save_period[index_bool2long_trt(save_period > 0)] - torch.ones_like(save_period[index_bool2long_trt(save_period > 0)])
        save_period[saved_idxes] = torch.ones_like(save_period[saved_idxes])*self.save_period

        saved_embed = embed[saved_idxes]
        prev_embed = mem_bank[saved_idxes]
        save_embed = self.save_proj(saved_embed)
        mem_padding_mask[saved_idxes] = torch.cat([mem_padding_mask[saved_idxes, 1:],
                                                   torch.zeros((mem_padding_mask[saved_idxes, 1:].shape[0], 1),
                                                               dtype=torch.int, device=embed.device)], dim=1)
        mem_bank = mem_bank.clone()
        mem_bank[saved_idxes] = torch.cat([prev_embed[:, 1:], save_embed], dim=1)
        return mem_padding_mask, save_period, mem_bank

    def _forward_temporal_attn_trt(self, mem_padding_mask, output_embedding, mem_bank):

        key_padding_mask = mem_padding_mask

        valid_idxes = index_bool2long_trt(key_padding_mask[:, -1] == 0)
        embed = output_embedding[valid_idxes]  # (n, 256)
        prev_embed = mem_bank[valid_idxes]
        key_padding_mask = key_padding_mask[valid_idxes]
        embed2 = self.temporal_attn(
            embed[None],
            prev_embed.transpose(0, 1),
            prev_embed.transpose(0, 1),
            key_padding_mask=key_padding_mask.bool(), # must be bool not int/long
        )[0][0]

        embed = self.temporal_norm1(embed + embed2)
        embed2 = self.temporal_fc2(F.relu(self.temporal_fc1(embed)))
        embed = self.temporal_norm2(embed + embed2)
        output_embedding = output_embedding.clone()
        output_embedding[valid_idxes] = embed

        return output_embedding

    def forward_trt(self, mem_padding_mask, output_embedding, mem_bank, scores, save_period):
        output_embedding = self._forward_temporal_attn_trt(mem_padding_mask, output_embedding, mem_bank)
        mem_padding_mask, save_period, mem_bank = self.update_trt(output_embedding, scores, mem_padding_mask, save_period, mem_bank)
        return output_embedding, mem_padding_mask, save_period, mem_bank

# QIM
class QueryInteractionBase(nn.Module):

    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__()
        self.args = args
        self._build_layers(args, dim_in, hidden_dim, dim_out)
        self._reset_parameters()

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        raise NotImplementedError()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _select_active_tracks(self, data: dict) -> Instances:
        raise NotImplementedError()

    def _update_track_embedding(self, track_instances):
        raise NotImplementedError()

class QueryInteractionModule(QueryInteractionBase):

    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__(args, dim_in, hidden_dim, dim_out)
        self.random_drop = args["random_drop"]
        self.fp_ratio = args["fp_ratio"]
        self.update_query_pos = args["update_query_pos"]

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        dropout = args["merger_dropout"]

        self.self_attn = nn.MultiheadAttention(dim_in, 8, dropout)
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim_in)

        if args["update_query_pos"]:
            self.linear_pos1 = nn.Linear(dim_in, hidden_dim)
            self.linear_pos2 = nn.Linear(hidden_dim, dim_in)
            self.dropout_pos1 = nn.Dropout(dropout)
            self.dropout_pos2 = nn.Dropout(dropout)
            self.norm_pos = nn.LayerNorm(dim_in)

        self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
        self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(dim_in)

        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def _update_track_embedding(self, track_instances: Instances) -> Instances:
        if len(track_instances) == 0:
            return track_instances
        dim = track_instances.query.shape[1]
        out_embed = track_instances.output_embedding
        query_pos = track_instances.query[:, :dim // 2]
        query_feat = track_instances.query[:, dim // 2:]
        q = k = query_pos + out_embed

        # attention
        tgt = out_embed
        tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:,
                                                                             0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        if self.update_query_pos:
            # ffn: linear_pos2
            query_pos2 = self.linear_pos2(
                self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
            query_pos = query_pos + self.dropout_pos2(query_pos2)
            query_pos = self.norm_pos(query_pos)
            track_instances.query[:, :dim // 2] = query_pos

        query_feat2 = self.linear_feat2(
            self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query_feat = query_feat + self.dropout_feat2(query_feat2)
        query_feat = self.norm_feat(query_feat)
        track_instances.query[:, dim // 2:] = query_feat
        # track_instances.ref_pts = inverse_sigmoid(track_instances.pred_boxes[:, :2].detach().clone())
        # update ref_pts using track_instances.pred_boxes
        return track_instances

    def _random_drop_tracks(self, track_instances: Instances) -> Instances:
        drop_probability = self.random_drop
        if drop_probability > 0 and len(track_instances) > 0:
            keep_idxes = torch.rand_like(track_instances.scores) > drop_probability
            track_instances = track_instances[keep_idxes]
        return track_instances

    def _add_fp_tracks(self, track_instances: Instances,
                       active_track_instances: Instances) -> Instances:
        """
        self.fp_ratio is used to control num(add_fp) / num(active)
        """
        inactive_instances = track_instances[track_instances.obj_idxes < 0]

        # add fp for each active track in a specific probability.
        fp_prob = torch.ones_like(
            active_track_instances.scores) * self.fp_ratio
        selected_active_track_instances = active_track_instances[
            torch.bernoulli(fp_prob).bool()]
        num_fp = len(selected_active_track_instances)

        if len(inactive_instances) > 0 and num_fp > 0:
            if num_fp >= len(inactive_instances):
                fp_track_instances = inactive_instances
            else:
                # randomly select num_fp from inactive_instances
                # fp_indexes = np.random.permutation(len(inactive_instances))
                # fp_indexes = fp_indexes[:num_fp]
                # fp_track_instances = inactive_instances[fp_indexes]

                # v2: select the fps with top scores rather than random selection
                fp_indexes = torch.argsort(inactive_instances.scores)[-num_fp:]
                fp_track_instances = inactive_instances[fp_indexes]

            merged_track_instances = Instances.cat(
                [active_track_instances, fp_track_instances])
            return merged_track_instances

        return active_track_instances

    def _select_active_tracks(self, data: dict) -> Instances:
        track_instances: Instances = data["track_instances"]
        if self.training:
            active_idxes = (track_instances.obj_idxes >=
                            0) & (track_instances.iou > 0.5)
            active_track_instances = track_instances[active_idxes]
            # set -2 instead of -1 to ensure that these tracks will not be selected in matching.
            active_track_instances = self._random_drop_tracks(
                active_track_instances)
            if self.fp_ratio > 0:
                active_track_instances = self._add_fp_tracks(
                    track_instances, active_track_instances)
        else:
            active_track_instances = track_instances[
                track_instances.obj_idxes >= 0]

        return active_track_instances

    def forward(self, data) -> Instances:
        active_track_instances = self._select_active_tracks(data)
        active_track_instances = self._update_track_embedding(
            active_track_instances)
        init_track_instances: Instances = data["init_track_instances"]
        merged_track_instances = Instances.cat(
            [init_track_instances, active_track_instances])
        return merged_track_instances


class QueryInteractionModuleTRTP(QueryInteractionModule):

    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super(QueryInteractionModule, self).__init__(args, dim_in, hidden_dim, dim_out)
        self.random_drop = args["random_drop"]
        self.fp_ratio = args["fp_ratio"]
        self.update_query_pos = args["update_query_pos"]

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        dropout = args["merger_dropout"]

        self.self_attn = nn.MultiheadAttention(dim_in, 8, dropout)
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim_in)

        if args["update_query_pos"]:
            self.linear_pos1 = nn.Linear(dim_in, hidden_dim)
            self.linear_pos2 = nn.Linear(hidden_dim, dim_in)
            self.dropout_pos1 = nn.Dropout(dropout)
            self.dropout_pos2 = nn.Dropout(dropout)
            self.norm_pos = nn.LayerNorm(dim_in)

        self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
        self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(dim_in)

        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def _update_track_embedding_trt(self, track_instances):
        dim = 512 #track_instances[0].shape[1]
        out_embed = track_instances[2]
        query_pos = track_instances[0][:, :dim // 2]
        query_feat = track_instances[0][:, dim // 2:]
        q = query_pos + out_embed
        k = query_pos + out_embed

        # attention
        tgt = out_embed
        tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:,
                                                                             0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        if self.update_query_pos:
            query_pos2 = self.linear_pos2(
                self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
            query_pos = query_pos + self.dropout_pos2(query_pos2)
            query_pos = self.norm_pos(query_pos)
            track_instances[0][:, :dim // 2] = query_pos

        query_feat2 = self.linear_feat2(
            self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query_feat = query_feat + self.dropout_feat2(query_feat2)
        query_feat = self.norm_feat(query_feat)
        track_instances[0][:, dim // 2:] = query_feat

        return track_instances

    def _select_active_tracks_trt(self, track_instances):
        active_index = index_bool2long_trt(track_instances[3] >= 0)
        active_track_instances = []
        for item in track_instances:
            active_track_instances.append(item[active_index])

        return active_track_instances

    def forward_trt(self, init_track_instances, track_instances):
        active_track_instances = self._update_track_embedding_trt(
            self._select_active_tracks_trt(track_instances))
        merged_track_instances = []
        for i in range(len(track_instances)):
            merged_track_instances.append(torch.cat((init_track_instances[i],
                                                     active_track_instances[i]), dim=0))
        return merged_track_instances