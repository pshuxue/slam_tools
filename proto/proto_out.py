from proto.slam_proto_cloud_pb2 import Intrinsic, RigCamera, Tbc
import google.protobuf as proto


rig = RigCamera()
rig.num_cam = 2
intri_tmp = rig.intrinsics.add()
intri_tmp.cols = 1
intri_tmp.rows = 4
intri_tmp.data.append(1)
intri_tmp.data.append(1)
intri_tmp.data.append(1)
intri_tmp.data.append(1)


T_tmp = rig.Tbcs.add()
T_tmp.cols = 4
T_tmp.rows = 4
for i in range(0, 4):
    for j in range(4):
        T_tmp.data.append(i*4+j)


fout = open("data.proto", 'wb')
tmp = rig.SerializeToString()
fout.write(tmp)
fout.close()
