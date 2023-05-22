BUNDLETRACK_DIR="/home/kausik/Documents/BundleTrack2.0"
NOCS_DIR="/media/bowen/e25c9489-2f57-42dd-b076-021c59369fec/DATASET/NOCS"
YCBINEOAT_DIR="/iros_submission_version"
CUSTOM_DIR="/home/kausik/Documents/BundleTrack2.0/Data"
echo "BUNDLETRACK_DIR $BUNDLETRACK_DIR"
echo "NOCS_DIR $NOCS_DIR"
echo "YCBINEOAT_DIR $YCBINEOAT_DIR"

# docker run --gpus all -it --network=host --name bundletrack  -m  16000m --cap-add=SYS_PTRACE --security-opt seccomp=unconfined  -v $BUNDLETRACK_DIR:$BUNDLETRACK_DIR:rw -v $NOCS_DIR:$NOCS_DIR -v $YCBINEOAT_DIR:$YCBINEOAT_DIR -v /tmp:/tmp  --ipc=host -e DISPLAY=${DISPLAY} -e GIT_INDEX_FILE wenbowen123/bundletrack:3090 bash
docker run --gpus all -it --network=host --name bundletrack2.0  -m  16000m --cap-add=SYS_PTRACE --security-opt seccomp=unconfined  -v $BUNDLETRACK_DIR:$BUNDLETRACK_DIR:rw -v $NOCS_DIR:$NOCS_DIR -v $YCBINEOAT_DIR:$YCBINEOAT_DIR -v $CUSTOM_DIR:$CUSTOM_DIR -v /tmp:/tmp  --ipc=host -e DISPLAY=${DISPLAY} -e GIT_INDEX_FILE wenbowen123/bundletrack:3090 bash
