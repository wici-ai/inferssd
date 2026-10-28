# SPDX-License-Identifier: Apache-2.0
"""Small deterministic integration fixture. No model-quality/performance claims."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'gguf-py'))
import gguf
out=Path(sys.argv[1] if len(sys.argv)>1 else 'qwen4exp-fixture.gguf')
w=gguf.GGUFWriter(out,'qwen4exp')
ints={'block_count':4,'context_length':128,'embedding_length':512,'vocab_size':256,
'attention.head_count':4,'attention.head_count_kv':1,'attention.key_length':32,'attention.value_length':32,
'expert_count':12,'expert_used_count':10,'expert_feed_forward_length':64,'expert_shared_feed_forward_length':64,
'ssm.conv_kernel':4,'ssm.state_size':32,'ssm.group_count':2,'ssm.time_step_rank':6,'ssm.inner_size':192,
'full_attention_interval':4,'rope.dimension_count':32,'hyper_connection.count':4,'hyper_connection.low_rank':32,
'attention.indexer.head_count':4,'attention.indexer.key_length':32,'attention.indexer.top_k':4,
'ple.ngram_size':3,'ple.heads_per_ngram':8,'ple.conv_kernel':4,'ple.eos_token_id':255,'embedding_length_per_layer_input':32}
for k,v in ints.items():w.add_uint32('qwen4exp.'+k,v)
if '--recurrent-array' in sys.argv[2:]:
 w.add_key_value('qwen4exp.attention.recurrent_layers',[True,True,True,False],gguf.GGUFValueType.ARRAY,gguf.GGUFValueType.BOOL)
w.add_float32('qwen4exp.attention.layer_norm_rms_epsilon',1e-6);w.add_float32('qwen4exp.rope.freq_base',1e7)
w.add_string('tokenizer.ggml.model','no_vocab');w.add_string('general.name','Qwen4Exp integration fixture (not model weights)')
mods=[1009,1013,1019,1021,1031,1033,1039,1049,1051,1061,1063,1069,1087,1091,1093,1097];offsets=np.cumsum([0]+mods[:-1]).tolist();rows=sum(mods)
for k,v in {'rope.dimension_sections':[6,5,5,0],'attention.compress_ratios':[0,0,0,4],'ple.layers':[1]}.items():w.add_key_value('qwen4exp.'+k,v,gguf.GGUFValueType.ARRAY,gguf.GGUFValueType.UINT32)
for k,v in {'ple.head_vocab_sizes':mods,'ple.head_offsets':offsets,'ple.layer_multipliers':[23703573157769,20109073645365,8052911324071]}.items():w.add_key_value('qwen4exp.'+k,v,gguf.GGUFValueType.ARRAY,gguf.GGUFValueType.UINT64)
rng=np.random.default_rng(20261004);manifest=[]
def tensor(name,shape,kind=gguf.GGMLQuantizationType.Q8_0,norm=False):
 a=rng.normal(0,0.025,tuple(reversed(shape))).astype(np.float32)
 if norm:a+=1
 if name.endswith('ssm_a'):a=-np.exp(a)
 if shape[0]%32 and kind==gguf.GGMLQuantizationType.Q8_0:kind=gguf.GGMLQuantizationType.F32
 if kind!=gguf.GGMLQuantizationType.F32:a=gguf.quants.quantize(a,kind)
 w.add_tensor(name,a,raw_dtype=kind)
 manifest.append({'name':name,'shape':shape,'type':kind.name})
def f32(name,shape,norm=False):tensor(name,shape,gguf.GGMLQuantizationType.F32,norm)
tensor('token_embd.weight',[512,256]);tensor('output.weight',[512,256]);f32('output_hc_norm.weight',[2048],True);tensor('output_hc_down.weight',[2048,32]);tensor('output_hc_up.weight',[32,2048]);tensor('per_layer_token_embd.weight',[32,rows])
for i in range(4):
 p=f'blk.{i}.'
 for branch in ['attn','ffn']:
  f32(p+f'hc_{branch}_norm.weight',[2048],True);tensor(p+f'hc_{branch}_down.weight',[2048,32]);tensor(p+f'hc_{branch}_up.weight',[32,2048]);tensor(p+f'hc_{branch}_inject.weight',[2048,4])
 if i==3:
  tensor(p+'attn_q.weight',[512,256]);tensor(p+'attn_k.weight',[512,32]);tensor(p+'attn_v.weight',[512,32]);tensor(p+'attn_output.weight',[128,512]);f32(p+'attn_q_norm.weight',[32],True);f32(p+'attn_k_norm.weight',[32],True)
  tensor(p+'indexer.q_proj.weight',[512,128],gguf.GGMLQuantizationType.BF16);tensor(p+'indexer.k_proj.weight',[512,32],gguf.GGMLQuantizationType.BF16);f32(p+'indexer.q_norm.weight',[32],True);f32(p+'indexer.k_norm.weight',[32],True)
 else:
  tensor(p+'attn_qkv.weight',[512,320]);tensor(p+'attn_gate.weight',[512,192]);f32(p+'ssm_conv1d.weight',[4,320]);f32(p+'ssm_dt.bias',[6]);f32(p+'ssm_a',[6]);tensor(p+'ssm_beta.weight',[512,6]);tensor(p+'ssm_alpha.weight',[512,6]);f32(p+'ssm_norm.weight',[32],True);tensor(p+'ssm_out.weight',[192,512])
 if i==1:
  tensor(p+'ple_key.weight',[512,2048]);tensor(p+'ple_value.weight',[512,512]);
  for n in ['key','query','conv']:f32(p+'ple_norm_'+n+'.weight',[2048],True)
  f32(p+'ple_conv1d.weight',[4,2048])
 tensor(p+'ffn_gate_inp.weight',[512,12]);tensor(p+'ffn_gate_exps.weight',[512,64,12]);tensor(p+'ffn_up_exps.weight',[512,64,12]);tensor(p+'ffn_down_exps.weight',[64,512,12]);tensor(p+'ffn_gate_inp_shexp.weight',[512]);tensor(p+'ffn_gate_shexp.weight',[512,64]);tensor(p+'ffn_up_shexp.weight',[512,64]);tensor(p+'ffn_down_shexp.weight',[64,512])
w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close();out.with_suffix('.json').write_text(json.dumps(manifest,indent=2));print(out, out.stat().st_size)
