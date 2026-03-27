from itertools import product
import os
import shutil


tcu = ["_tCu10um", "_tCu20um"]
wcu = ["_wCu200um", "_wCu250um"]
tlam = ["_tLamCore100nm", "_tLamCore350nm"]
nlam = ["_Nlam8", "_Nlam16"]
taln = ["_tAlN15nm", "_tAlN20nm"]
tsu8 = ["_tSu82nm", "_tSu86nm"]

valid_names = ["Z9477"+ a + b + c + d +e +f +"_RLQ" for a, b,c,d,e,f in product(tcu, wcu, tlam, nlam, taln, tsu8)]


src_folder  = "RLQ/pinn_RLQ"
dst_folder = "RLQ/inter"

# os.makedirs(dst_folder, exist_ok=True)
# for filename in os.listdir(src_folder):
#     src_path = os.path.join(src_folder, filename)
#     if os.path.isfile(src_path):
#         name, ext = os.path.splitext(filename)   # 去掉后缀再比较
#         if name in valid_names:
#             shutil.copy2(src_path, os.path.join(dst_folder, filename))
#             print(f"Copied: {filename}")
#

for filename in os.listdir(src_folder):
    src_path = os.path.join(src_folder, filename)
    if os.path.isfile(src_path):
        name, ext = os.path.splitext(filename)   # 去掉后缀再比较
        if name in valid_names:
            valid_names.remove(name)
for names in valid_names:
    print(names)