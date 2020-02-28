import pickle
import matplotlib.pyplot as plt


l=pickle.load(open("data/train_ffts/crossvalid/ffts_crossvalid.p", "rb"))
plt.figure(figsize=(30,30))
plt.plot(l[100][0:8000])
plt.ylim(0,0.2)
plt.xlim(0,1000)
plt.savefig("test.png")
