import torch
from torch import nn
import torch.nn.functional as F
from .weakly_models import Visual_Attention,Audio_Visual_Fusion_Attention
from .weakly_models import EncoderLayer,EncoderLayer1, Encoder, DecoderLayer, Decoder
from torch.nn import MultiheadAttention


class InternalTemporalRelationModule(nn.Module):
    def __init__(self, input_dim, d_model):
        super(InternalTemporalRelationModule, self).__init__()
        self.encoder_layer = EncoderLayer(d_model=d_model, nhead=4)
        self.encoder_layer1 = EncoderLayer1(d_model=d_model, nhead=4)
        self.encoder = Encoder(self.encoder_layer, self.encoder_layer1,num_layers=2)

        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=True)
        # add relu here?

    def forward(self, feature):
        # feature: [seq_len, batch, dim]
        feature = self.affine_matrix(feature)
        feature = self.encoder(feature)

        return feature


class CrossModalRelationAttModule(nn.Module):
    def __init__(self, input_dim, d_model):
        super(CrossModalRelationAttModule, self).__init__()

        self.decoder_layer = DecoderLayer(d_model=d_model, nhead=4)
        self.decoder = Decoder(self.decoder_layer, num_layers=1)

        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, query_feature, memory_feature):
        query_feature = self.affine_matrix(query_feature)
        output = self.decoder(query_feature, memory_feature)

        return output


class WeaklyLocalizationModule(nn.Module):
    def __init__(self, input_dim):
        super(WeaklyLocalizationModule, self).__init__()

        self.hidden_dim = input_dim # need to equal d_model
        self.classifier = nn.Linear(self.hidden_dim, 1) # start and end
        self.event_classifier = nn.Linear(self.hidden_dim, 29)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, fused_content):
        fused_content = fused_content.transpose(0, 1)
        max_fused_content, _ = fused_content.max(1)
        # confident scores
        is_event_scores = self.classifier(fused_content)
        # classification scores
        raw_logits = self.event_classifier(max_fused_content)[:, None, :]
        # fused
        fused_logits = is_event_scores.sigmoid() * raw_logits
        # Training: max pooling for adapting labels
        logits, _ = torch.max(fused_logits, dim=1)
        event_scores = self.softmax(logits)

        return is_event_scores.squeeze(), raw_logits.squeeze(), event_scores

class ForwardBackwardFusionModule(nn.Module):
    def __init__(self, d_model, n_head, head_dropout=0.1):
        super(ForwardBackwardFusionModule, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.video_multihead = MultiheadAttention(d_model, num_heads=n_head, dropout=head_dropout)
        self.norm1 = nn.LayerNorm(d_model)


    def forward(self, video_feat, audio_feat):
        # video_feat, audio_feat: [10, batch, 256]
        global_feat = video_feat * audio_feat
        memory = torch.cat([audio_feat, video_feat], dim=0)
        mid_out = self.video_multihead(global_feat, memory, memory)[0]
        output = self.norm1(global_feat + self.dropout(mid_out))

        return  output

class weak_main_model(nn.Module):
    def __init__(self):
        super(weak_main_model, self).__init__()
        self.spatial_channel_att = Visual_Attention().cuda()
        self.avf_att = Audio_Visual_Fusion_Attention().cuda()
        self.video_input_dim = 512 
        self.video_fc_dim = 512
        self.d_model = 256
        self.v_fc = nn.Linear(self.video_input_dim, self.video_fc_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.av_encoder = InternalTemporalRelationModule(input_dim=512, d_model=256)
        self.av_audio_decoder = CrossModalRelationAttModule(input_dim=512, d_model=256)
        self.audio_encoder = InternalTemporalRelationModule(input_dim=128, d_model=256)
        self.audio_decoder = CrossModalRelationAttModule(input_dim=128, d_model=256)

        self.video_encoder = InternalTemporalRelationModule(input_dim=512, d_model=256)
        self.av_video_decoder = CrossModalRelationAttModule(input_dim=512, d_model=256)
        self.video_decoder = CrossModalRelationAttModule(input_dim=512, d_model=256)   

        self.FBFM1 = ForwardBackwardFusionModule(self.d_model, n_head=4, head_dropout=0.1)
        self.FBFM2 = ForwardBackwardFusionModule(256, n_head=4, head_dropout=0.1).cuda()        
        self.localize_module = WeaklyLocalizationModule(self.d_model)


    def forward(self, visual_feature, audio_feature):
        # [batch, 10, 512]
        # this fc is optinal, that is used for adaption of different visual features (e.g., vgg, resnet).
        audio_feature = audio_feature.transpose(1, 0).contiguous()
        visual_feature = self.v_fc(visual_feature)
        visual_feature = self.dropout(self.relu(visual_feature))

        # visual_attention         
        att_visual_feature = self.spatial_channel_att(visual_feature, audio_feature)
        # audio_visual_fusion_attention   
        av_feature = self.avf_att(visual_feature, audio_feature)

        # forward attention
        av_feature = av_feature.transpose(1, 0).contiguous()
        att_visual_feature = att_visual_feature.transpose(1, 0).contiguous()
        
        # backward attention
        av_key_value_feature = self.av_encoder(av_feature)
        aav = self.audio_decoder(audio_feature, av_key_value_feature)
        
        # backward attention
        audio_key_value_feature = self.audio_encoder(audio_feature)
        ava = self.av_audio_decoder(av_feature, audio_key_value_feature)

        # forward attention
        vdieo_key_value_feature = self.video_encoder(att_visual_feature)    
        avv = self.av_video_decoder(av_feature, vdieo_key_value_feature)

        vav = self.video_decoder(att_visual_feature, av_key_value_feature)
  
        #Forward Backward Fusion Module
        a= self.FBFM1(ava, aav)
        v= self.FBFM2(avv, vav)
        av = torch.mul(a+v,0.5)
        scores = self.localize_module(av)

        return scores
