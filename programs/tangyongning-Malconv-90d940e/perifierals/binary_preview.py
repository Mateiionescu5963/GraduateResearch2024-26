import pefile
from capstone import *

pe = pefile.PE("/home/dgy/Desktop/DikeDataset-main/DikeDataset-main/files/malware/00a0d8c3adc67e930fd89331e4e41cfe2a7128072d5d3ca0ec369da5b7847a45.exe")

entry_point = pe.OPTIONAL_HEADER.AddressOfEntryPoint
entry_point_address = entry_point+pe.OPTIONAL_HEADER.ImageBase

binary = pe.get_memory_mapped_image()[entry_point:entry_point+100]
disassembler = Cs(CS_ARCH_X86, CS_MODE_32)

for i in disassembler.disasm(binary, entry_point_address):
    print("0x%x:\t%s\t%s" %(i.address, i.mnemonic, i.op_str))