"""Module containing different model implementations."""
# from .attention_smoothing import SmoothedAttentionEncoder
from .gru_simple import GRUSimpleModel
from .gru_d import GRUDModel
from .interpolation_prediction import InterpolationPredictionModel
from .phased_lstm import PhasedLSTMModel
from .transformer import TransformerModel
from .model_asyncts_new import Classifier_RESNET_new
from .deep_set_attention import DeepSetAttentionModel
from .model_asyncts import Classifier_RESNET
from .resnet_simple_model import Classifier_RESNET_simple
from .resnet_simple_model_2 import Classifier_RESNET_simple_2
from .model_asyncts_att import Classifier_RESNET_att
from .model_asyncts_att_gap import Classifier_RESNET_att_gap
from .Set_cnn_att import Set_cnn_att
from .model_incep import Classifier_INCEP
from .model_asyncts_activity import Classifier_RESNET_act
from .model_asyncts_activity_simple import Classifier_RESNET_act_simple
from .model_ANN_activity import ANN_activity
from .model_asyncts_online import Classifier_RESNET_online
from .model_asyncts_att_new import Classifier_RESNET_att_new
from .resnet_mixer import Classifier_mixer
from .model_asyncts_nl import Classifier_RESNET_nl
from .model_asyncts_att_nl import Classifier_RESNET_att_nl
from .gru_asycnts import Classifier_GRU_nl
from .gru_asyncts_att_nl import Classifier_GRU_att_nl
from .resnet_act_simple_model import Simple_RESNET_act
from .model_asyncts_activity_att import Classifier_RESNET_act_att
from .model_resnet_mean import RESNET_mean
from .model_resnet_forward import RESNET_forward
from .model_attn import attn_nl
from .model_attn_time_nl import attn_time_nl
from .model_resnet_forward_act import RESNET_forward_act
from .model_resnet_mean_act import RESNET_mean_act
from .model_ablaction_resnet_sum_nl import ablation_RESNET_sum_nl
from .model_ablation_resnet_mean_act import ablation_RESNET_mean_act
# from .model_asyncts_activity_att2 import Classifier_RESNET_act_att
from .triple_cnn import triple_cnn
from .model_bi import bi_nl
from .model_atp import atp_nl
from .model_graph import graph
from .model_graph_sparse import graph_sparse
from .model_graph_act import graph_act
from .model_best_channel import best_channel
from .model_simple import simple
from .model_simple_time import simple_time
from .model_simple_time_pos import simple_time_pos
from .model_simple_time_diff import simple_time_diff
from .model_simple_batch import simple_time_batch
from .model_concat import concat
from .model_simple_time_mean import simple_time_mean
# __all__ = [
#     'GRUSimpleModel', 'PhasedLSTMModel', 'InterpolationPredictionModel',
#     'GRUDModel', 'TransformerModel', 'DeepSetAttentionModel'
# ]
__all__ = ['simple_time_mean','concat','simple_time_batch','simple_time_pos','simple_time_diff', 'simple_time','simple', 'best_channel','graph_act','graph_sparse','graph', 'atp_nl','bi_nl', 'triple_cnn','ablation_RESNET_mean_act','ablation_RESNET_sum_nl','RESNET_mean_act','RESNET_forward_act', 'attn_time_nl','attn_nl','RESNET_forward','RESNET_mean','Classifier_RESNET_act_att','Simple_RESNET_act','Classifier_GRU_att_nl','Classifier_GRU_nl','Classifier_RESNET_att_nl','Classifier_RESNET_nl','Classifier_mixer','Classifier_RESNET_att_new','Classifier_RESNET_online','ANN_activity','Classifier_RESNET_act_simple','Classifier_RESNET_act','Classifier_INCEP','Set_cnn_att', 'Classifier_RESNET_att_gap','Classifier_RESNET_att', 'Classifier_RESNET_new','Classifier_RESNET', 'InterpolationPredictionModel', 'DeepSetAttentionModel',  'GRUSimpleModel', 'GRUDModel', 'TransformerModel', 'Classifier_RESNET_simple', 'PhasedLSTMModel', 'Classifier_RESNET_simple_2']
