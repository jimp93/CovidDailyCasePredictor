t= 20
print(t.__class__.__bases__)
print(t.__class__)

o = object
o.__new__(int)