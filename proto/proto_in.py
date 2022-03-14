from proto.slam_proto_cloud_pb2 import Intrinsic, RigCamera
import google.protobuf as proto


fin = open("data.proto", 'rb')
lens = fin.readlines()
rig = RigCamera()

rig.ParseFromString(lens[0])


print(rig.intrinsics[0].data[2])

for i in range(16):
    print(rig.Tbcs[0].data[i])
# print(rig.intrinsics[1].cols)
