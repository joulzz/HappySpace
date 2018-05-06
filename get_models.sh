wget https://transfer.sh/ZIWAD/sentiment_net.data-00000-of-00001 https://transfer.sh/R9ZEw/sentiment_net.index https://transfer.sh/RdRbY/sentiment_net.meta https://transfer.sh/SAVbX/haarcascade_frontalface_default.xml  https://transfer.sh/miPqx/svm_model.pkl
mkdir Models
mkdir Models/sentiment_net
mv sentiment_net* Models/sentiment_net
mv haarcascade* Models/
mv svm_model.pkl Models/
