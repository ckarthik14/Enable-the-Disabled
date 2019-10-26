with open("charlotte.txt","r") as f:
    data=f.readlines()
    sent=''
    for dat in data:
        dat=dat[:-1]
        sent+=dat
with open("charlotte.txt","w") as f:
    f.writelines(sent)
    
    print(sent)