def xxmm(a):
    if isinstance(a, int) :
        a+=1
        a-=1
        t=a
        a*=2
        return t
    elif isinstance(a, list):
        a.append(("adsd","sdsasaddds"))
        b=1
        b*=2
        a.pop()
        return a
    elif isinstance(a, dict):
        a["aaa"]=9
        a["kkk"]=a["aaa"]
        b = 1
        b *= 2
        del a["aaa"]
        del a["kkk"]
        return a
    elif isinstance(a, float):
        t=a
        b=a*2
        a=t
        xxmm(int(b))
        return t
    elif isinstance(a, str):
        a+="str"
        b="123"
        xxmm(int(b))
        return a[0:-3]
a=111
a=xxmm(a)
print(a)