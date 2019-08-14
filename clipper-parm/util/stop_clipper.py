from clipper_admin import ClipperConnection, ParmDockerContainerManager
clipper_conn = ClipperConnection(ParmDockerContainerManager(), False)
clipper_conn.stop_all()
