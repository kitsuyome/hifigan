mkdir -p /kaggle/working/hifigan/src/data/testmels

curl -L "https://www.dropbox.com/scl/fi/k0dn6d2muijz8s2o5qisd/audio_1.pt?rlkey=44i6beg23zot3iztai0ypp3w7&dl=0" -o /kaggle/working/hifigan/src/data/testmels/audio_1.pt
curl -L "https://www.dropbox.com/scl/fi/fgqaqr6jnx544ez9ufbs5/audio_2.pt?rlkey=w3tgwuhtg5ld4pxja9dlx9tvc&dl=0" -o /kaggle/working/hifigan/src/data/testmels/audio_2.pt
curl -L "https://www.dropbox.com/scl/fi/wu3j6l3n8w1omtronlz4u/audio_3.pt?rlkey=4el8wmfjnw288r8d30f7pgkqg&dl=0" -o /kaggle/working/hifigan/src/data/testmels/audio_3.pt

mkdir -p /kaggle/working/checkpoints

wget "https://www.dropbox.com/scl/fi/oro4s8sdgu9sw5efyuc1r/checkpoint.pth?rlkey=uqc4qimpludv81wb54c19gxm0&dl=0" -O /kaggle/working/checkpoints/checkpoint.pth
