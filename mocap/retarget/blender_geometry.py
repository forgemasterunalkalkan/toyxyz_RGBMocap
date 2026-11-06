import bpy,json,numpy as np
from .mapping import A
from .file_paths import io_path
def register(OUT,kind,rig,arm,mapping,rest):
 mapped={name:i for i,name in enumerate(mapping)};bone_map={}
 if kind=='pmx':
  for side,suffix in enumerate(['.L','.R']):
   for stem,joint in [('足D',1+side),('ひざD',4+side),('足首D',7+side),('足先EX',10+side)]:
    mapped[stem+suffix]=joint
 for b in arm.data.bones:
  parent=b
  while parent and parent.name not in mapped:parent=parent.parent
  bone_map[b.name]=mapped[parent.name] if parent else 0
 vertices=[];weights=[];wrist_weights=[];mesh_names=[];mesh_counts=[]
 for obj in bpy.context.scene.objects:
  if obj.type!='MESH':continue
  mesh_names.append(obj.name);mesh_counts.append(len(obj.data.vertices))
  # Native outline duplicates are handled by Blender validation separately.
  M=np.array(obj.matrix_world);v=np.empty(len(obj.data.vertices)*3);obj.data.vertices.foreach_get('co',v);v=v.reshape(-1,3);vertices.append((v@M[:3,:3].T+M[:3,3])@A)
  w=np.zeros((len(v),22));hw=np.zeros((len(v),2))
  for vertex in obj.data.vertices:
   for group in vertex.groups:
    name=obj.vertex_groups[group.group].name
    if name in bone_map:w[vertex.index,bone_map[name]]+=group.weight
    for h in [0,1]:
     if name==mapping[20+h]:hw[vertex.index,h]+=group.weight
  total=w.sum(1);missing=total<1e-8;w[missing,0]=1.;w/=np.maximum(w.sum(1,keepdims=True),1e-8);weights.append(w);wrist_weights.append(hw)
 v=np.concatenate(vertices);W=np.concatenate(weights);HW=np.concatenate(wrist_weights);height=float(rest[:,1].max()-rest[:,1].min());payload=dict(rest=rest,mesh_names=mesh_names,mesh_counts=mesh_counts)
 counts=[]
 for side in [0,1]:
  foot=7+side;toe=10+side;shin=4+side;p=rest[foot];length=np.linalg.norm(rest[toe]-p);leg=np.linalg.norm(rest[foot]-rest[shin])+np.linalg.norm(rest[shin]-rest[1+side])
  mask=(W[:,[foot,toe,shin]].sum(1)>.65)&(v[:,1]<p[1]+.025*height)&(abs(v[:,0]-p[0])<max(.10*height,.12))&(abs(v[:,2]-p[2])<max(2.5*length,.22))
  ids=np.flatnonzero(mask);assert len(ids)>10,(kind,side,len(ids))
  ground=float(np.percentile(v[ids,1],1));bottom=ids[v[ids,1]<ground+max(.006,.004*height)];assert len(bottom)>=4
  forward=(rest[toe]-p)*[1,0,1];forward/=max(np.linalg.norm(forward),1e-8);along=(v[bottom]-p)@forward;split=float(np.median(along))
  heel=bottom[along<=split];fore=bottom[along>split]
  if len(heel)<2 or len(fore)<2:heel=bottom[:len(bottom)//2];fore=bottom[len(bottom)//2:]
  # Store registered material indices in the complete eligible-foot subset.
  lookup={value:i for i,value in enumerate(ids)}
  payload.update({f'foot_v{side}':v[ids],f'foot_w{side}':W[ids],f'foot_ids{side}':ids,f'heel{side}':np.array([lookup[i] for i in heel]),f'toe{side}':np.array([lookup[i] for i in fore])})
  wrist=20+side;hand=arm.data.bones[mapping[wrist]];finger=[b for b in arm.data.bones if hand in b.parent_recursive]
  # Registered palm region excludes finger-dominant vertices.
  palm_length=max(.025,float(np.median([np.linalg.norm(np.array(arm.matrix_world@b.head_local)@A-rest[wrist]) for b in finger if np.linalg.norm(np.array(arm.matrix_world@b.head_local)@A-rest[wrist])>.015])) if finger else .055)
  hand_dist=np.linalg.norm(v-rest[wrist],axis=1)
  hand_ids=np.flatnonzero((HW[:,side]>.95)&(hand_dist>palm_length*.15)&(hand_dist<min(palm_length*1.1,.14*height)))
  if len(hand_ids)<8:hand_ids=np.flatnonzero((HW[:,side]>.85)&(hand_dist<.10*height))
  # Rigs that share palm skin between hand and finger bones (hand weight below 0.85).
  if len(hand_ids)<=3:hand_ids=np.flatnonzero((HW[:,side]>.5)&(hand_dist<min(palm_length*1.1,.14*height)))
  assert len(hand_ids)>3,(kind,side,'palm')
  payload.update({f'palm_v{side}':v[hand_ids],f'palm_w{side}':W[hand_ids],f'palm_ids{side}':hand_ids})
  counts.append(dict(foot_vertices=len(ids),sole_vertices=len(bottom),palm_vertices=len(hand_ids),shin_weighted_sole_vertices=int((W[bottom,shin]>.5).sum())))
 np.savez(io_path(OUT/(kind+'-contact-geometry.npz')),**payload)
