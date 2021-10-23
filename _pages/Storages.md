# Storages

다양한 Storage에 대해서 일단 정리



|           | Block                  | File        | Object          |
| --------- | ---------------------- | ----------- | --------------- |
| 일관성    | 강한 일관성            | 강한 일관성 | 궁극적인 일관성 |
| 구조      | 블록수준 고도로 구조화 | 계층 구조화 | 비구조화        |
| 접근 수준 | Block 수준             | 파일 수준   | Object 수준     |



## Block Storage

- format, file-system 생성 후 사용하는 가장 기본적인 스토리지
- typical native storage interface of most storage media at the driver level
- block storage를 마운트 된 장비에서 여러 용도로 나눠서 사용 
  - ex) DBMS, NFS, HDFS, etc

- latency가 낮음
- aws ebs 최초 mount

```
sudo mkfs -t xfs /dev/xvfs      // mkfs.xfs /dev/xvfs 도 가능
mkdir /data
mount /dev/xvfs /data
```

- aws ebs resize

```
1) 콘솔이나 명령어로 block size 증가시킴
2) ec2에서 명령어 수행
  # lsblk  
  2-1) partition 있는 block일 경우
  # growpart /dev/xvda 1
  # lsblk
  # df -hT
  // xfs 파일시스템이고 /에 마운트되어 있을 경우
  # xfs_growfs -d /
  2-2) partition 없는 block일 경우
  # xfs_growfs -d /ebs
```

|                    | AWS  | GCP                        | Azure |
| ------------------ | ---- | -------------------------- | ----- |
| Block Storage      | EBS  | Persistent Disk<br />SSD   | Disk  |
| Object Storage     | S3   | Google Cloud Storage (GCS) | Blob  |
| File share Storage | EFS  | Filestore                  | File  |



## File Storage

- 파일시스템으로 구성된 스토리지
- folder 구조로 관리
- metadata 정보는 비교적 적음 (생성일, 수정일, 파일사이즈 등)
- 데이터가 많아지면서 성능 이슈 발생 가능
- 일반적 하드 드라이브, NFS(NAS), CIFS 등
- aws efs mount

```
# 사전에 efs 용 sg 생성 (ec2 sg에서 tcp 2049 port listen 가능하도록 설정)
sudo yum install nfs-utils
sudo mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport 172.31.8.164:/ efs
```



## Object Storage

- object 단위로 저장/관리
- block이나 folder가 아닌 단일 repository에 데이터 저장. 평면주소공간에 저장
- object name : key 역할
- object는 metadata 가짐 
  - 사용기간, 보안, access policy, etc
- HTTP API로 access
- 수정 불가



### FileSystem Formats

- 고려해야하는 부분: scalability, stability, data integrity

|      | EXT4 (Extended filesystem 4th gen)                           | XFS                                        | NTFS                | ZFS   | BTRFS |
| ---- | ------------------------------------------------------------ | ------------------------------------------ | ------------------- | ----- | ----- |
| OS   | Linux                                                        | Linux                                      | Windows             | Linux | Linux |
| 특징 | 많이 사용됨<br />Debian, Ubuntu 기반 standard fs             | 많이 사용됨<br />RHEL 기반 OS의 default fs | Dominant filesystem |       |       |
|      | 2008년 출시<br />many small files 관리<br />write cache 중단 시에도 metadata 변경 보장<br /> - max file size:16TiB까지 <br />- Backward compatibility with Ext3, Ext2<br />- Storage block allocation 효율성 향상 (read/write 성능 향상)<br />- journal checksum& faster fs checks<br />- timestamp in nanoseconds<br />- sub-directory limit 없음 (HTreeindices 사용)<br />- Transparent encryption 적용 | - max file size: 8XB                       |                     |       |       |



#### Commands

```
df -h
df -hT
lsblk
```







###### References

[object, block, file storage 차이]: https://www.alibabacloud.com/ko/knowledge/difference-between-object-storage-file-storage-block-storage	"object, file, block storage 차이점"
[Block Storage]: https://www.delltechnologies.com/ko-kr/learn/data-storage/block-storage.htm
[ext4 vs xfs]: https://linoxide.com/ext4-vs-xfs-which-one-to-choose/

