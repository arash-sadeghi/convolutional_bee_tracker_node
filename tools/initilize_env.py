import os,sys

from numpy import extract

def checkAndCreateFolder(in_path):
    if not os.path.isdir(in_path):
        os.makedirs(in_path,exist_ok=True)

def main():
    download_link="https://owncloud.roboroyale.eu/s/vpN28Nw5Uy4omn9/download"
    current_path=os.path.dirname(__file__)
    os.chdir(current_path+"/..")

    out_path="data"
    checkAndCreateFolder(out_path)
    os.chdir(out_path)

    essential_folder="essentials_"
    checkAndCreateFolder(essential_folder)
    os.chdir(essential_folder)
    os.system("wget {}".format(download_link))
    zip_name="download"
    os.system("unzip {}".format(zip_name))
    if "used_kernel.npy" in os.listdir():
        print("[+] extract successful")
    else:
        print("[-] extraction failed")

if __name__=='__main__':
    main()
