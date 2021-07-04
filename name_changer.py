import os
for count, filename in enumerate(os.listdir("./data/real")):
    dst = "real_" + str(count) + ".jpg"
    src = "./data/real/" + filename
    dst = "./data/real/" + dst
        
    # rename() function will
    # rename all the files
    os.rename(src, dst)

for count, filename in enumerate(os.listdir("./data/fake")):
    dst = "fake_" + str(count) + ".jpg"
    src = "./data/fake/" + filename
    dst = "./data/fake/" + dst
        
    # rename() function will
    # rename all the files
    os.rename(src, dst)