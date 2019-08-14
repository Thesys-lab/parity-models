import struct

WORKER_MGMT_PORT = 8898
WORKER_RECV_PORT = 8899

# |worker_idx| |mb_to_send|
master_to_snd_format_struct = struct.Struct('I I')

snd_to_recv_format_struct = struct.Struct('I')
