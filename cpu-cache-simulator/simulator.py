import argparse
import random
import util
from cache import Cache
from memory import Memory


def read(address, memory, cache):
    """Read a byte from cache."""
    cache_block = cache.read(address)

    if cache_block:
        global hits
        hits += 1
    else:
        block = memory.get_block(address)
        victim_info = cache.load(address, block)
        cache_block = cache.read(address)

        global misses
        misses += 1

        # Write victim line's block to memory if replaced
        if victim_info:
            memory.set_block(victim_info[0], victim_info[1])

    return cache_block[cache.get_offset(address)]


def write(address, byte, memory, cache):
    """Write a byte to cache."""
    written = cache.write(address, byte)

    if written:
        global hits
        hits += 1
    else:
        global misses
        misses += 1

    if args.WRITE == Cache.WRITE_THROUGH:
        # Write block to memory
        block = memory.get_block(address)
        block[cache.get_offset(address)] = byte
        memory.set_block(address, block)
    elif args.WRITE == Cache.WRITE_BACK:
        if not written:
            # Write block to cache
            block = memory.get_block(address)
            cache.load(address, block)
            cache.write(address, byte)


replacement_policies = ["LRU", "LFU", "FIFO", "RAND"]
write_policies = ["WB", "WT"]

parser = argparse.ArgumentParser(description="Simulate the cache of a CPU.")

parser.add_argument("MEMORY", metavar="MEMORY", type=int,
                    help="Size of main memory in 2^N bytes")
parser.add_argument("CACHE", metavar="CACHE", type=int,
                    help="Size of the cache in 2^N bytes")
parser.add_argument("BLOCK", metavar="BLOCK", type=int,
                    help="Size of a block of memory in 2^N bytes")
parser.add_argument("MAPPING", metavar="MAPPING", type=int,
                    help="Mapping policy for cache in 2^N ways")
parser.add_argument("REPLACE", metavar="REPLACE", choices=replacement_policies,
                    help="Replacement policy for cache {"+", ".join(replacement_policies)+"}")
parser.add_argument("WRITE", metavar="WRITE", choices=write_policies,
                    help="Write policy for cache {"+", ".join(write_policies)+"}")

args = parser.parse_args()

mem_size = 2 ** args.MEMORY
cache_size = 2 ** args.CACHE
block_size = 2 ** args.BLOCK
mapping = 2 ** args.MAPPING

hits = 0
misses = 0

global global_counter
global_counter = 1
global global_dict
global_dict = {}
global addr_reuse
addr_reuse = {}
global addr_lastUsage
addr_lastUsage = {}

def getAddr(line):
    global global_counter
    global global_dict
    tmp = []
    for item in line:
        if item != " " and item != "<" and item != ">":
            tmp.append(item)
    if len(tmp) == 0:
        return -1
    addrStr = tmp[0]
    addr = 0
    if addrStr in global_dict.keys():
        return global_dict[addrStr] * 8
    else:
        global_dict[addrStr] = global_counter
        addr = global_counter
        global_counter += 1
        return addr * 8

memory = Memory(mem_size, block_size)
cache = Cache(cache_size, mem_size, block_size,
              mapping, args.REPLACE, args.WRITE)

mapping_str = "2^{0}-way associative".format(args.MAPPING)
print("\nMemory size: " + str(mem_size) +
      " bytes (" + str(mem_size // block_size) + " blocks)")
print("Cache size: " + str(cache_size) +
      " bytes (" + str(cache_size // block_size) + " lines)")
print("Block size: " + str(block_size) + " bytes")
print("Mapping policy: " + ("direct" if mapping == 1 else mapping_str) + "\n")

command = None

while (command != "quit"):
    operation = input("> ")
    operation = operation.split()

    try:
        command = operation[0]
        params = operation[1:]

        if command == "read" and len(params) == 1:
            address = int(params[0])
            byte = read(address, memory, cache)

            print("\nByte 0x" + util.hex_str(byte, 2) + " read from " +
                  util.bin_str(address, args.MEMORY) + "\n")

        elif command == "write" and len(params) == 2:
            address = int(params[0])
            byte = int(params[1])

            write(address, byte, memory, cache)

            print("\nByte 0x" + util.hex_str(byte, 2) + " written to " +
                  util.bin_str(address, args.MEMORY) + "\n")

        elif command == "randread" and len(params) == 1:
            amount = int(params[0])

            for i in range(amount):
                address = random.randint(0, mem_size - 1)
                read(address, memory, cache)

            print("\n" + str(amount) + " bytes read from memory\n")

        elif command == "randwrite" and len(params) == 1:
            amount = int(params[0])

            for i in range(amount):
                address = random.randint(0, mem_size - 1)
                byte = util.rand_byte()
                write(address, byte, memory, cache)

            print("\n" + str(amount) + " bytes written to memory\n")

        elif command == "printcache" and len(params) == 2:
            start = int(params[0])
            amount = int(params[1])

            cache.print_section(start, amount)

        elif command == "printmem" and len(params) == 2:
            start = int(params[0])
            amount = int(params[1])

            memory.print_section(start, amount)

        elif command == "stats" and len(params) == 0:
            ratio = (hits / ((hits + misses) if misses else 1)) * 100

            print("\nHits: {0} | Misses: {1}".format(hits, misses))
            print("Hit/Miss Ratio: {0:.2f}%".format(ratio) + "\n")

        elif command == "ptd":
            # PTD = prepare training data
            # Param 0 = file name
            # Param 1 = look-ahead window
            content = []
            if(len(params) == 2):
                content = f.readlines()
                content = [x.strip().split(" ") for x in content]


            if(len(params) == 3):
                with open(params[0]) as f:
                    for i in range(int(params[2])):
                        content.append(f.readline().strip().split(" "))

            # print("CONTENT: ", len(content))
            # print("EX: ", content)
            for i in range(len(content)):
                try:
                    line = content[i]
                    if line[0] != "<" and line[0] != ">":
                        continue
                    addr = getAddr(line)
                    # print("ADDR: ", addr)
                    if addr == -1:
                        continue
                    # print("ADDR: ", addr)
                    # First do operation; assume reads as we don't differentiate between reads & writes
                    byte = read(addr, memory, cache)
                    # print("\nByte 0x" + util.hex_str(byte, 2) + " read from " +
                    #     util.bin_str(addr, args.MEMORY) + "\n")
                    # Then calculate miss rate at this moment
                    missRate = (misses / ((hits + misses) if misses else 1))

                    # Reuse count & distance are also features
                    # pastReuseCount = 0
                    # pastReuseDistance = 0
                    # for j in range(i-1, max(i-int(params[1]), -1), -1):
                    #     addr2 = getAddr(content[j])
                    #     if addr2 == -1:
                    #         continue
                    #     if addr == addr2:
                    #         pastReuseCount += 1
                    #         if pastReuseDistance == 0:
                    #             pastReuseDistance = i - j

                    pastReuse = addr_lastUsage.get(addr, i)
                    addr_lastUsage[addr] = i
                    pastReuseDistance = i - pastReuse

                    pastReuseCount = addr_reuse.get(addr, 0)
                    addr_reuse[addr] = pastReuseCount + 1

                    # Calculate FUTURE REUSE label (reused = 1, not reused = 0) with lookahead window given as parameter
                    reused = 0
                    endSearch = min(len(content) - 1, i + int(params[1]))
                    for j in range(min(i+1, endSearch), endSearch):
                        addr2 = getAddr(content[j])
                        if addr2 == -1:
                            continue
                        if addr == addr2:
                            reused = 1
                            break

                    line.append(str(missRate))
                    line.append(str(pastReuseCount))
                    line.append(str(pastReuseDistance))
                    line.append(str(reused))
                    # Now write to new file
                    # print("LINE: ", line)
                except Exception as e:
                    print("\nERROR: ", e)
            with open("res.txt", "w") as f:
                for line in content:
                    try:
                        if len(line) == 6:
                            line.pop(0)
                            line[0] = line[0].strip()
                            # print(" ".join(line) + "\n")
                            f.write(" ".join(line) + "\n")
                    except Exception as e:
                        print("\nERROR: ", e)

            print("DONE WRITING")
            ratio = (hits / ((hits + misses) if misses else 1)) * 100
            print("\nHits: {0} | Misses: {1}".format(hits, misses))
            print("Hit/Miss Ratio: {0:.2f}%".format(ratio) + "\n")


        elif command != "quit":
            print("\nERROR: invalid command\n")

    except IndexError:
        print("\nERROR: out of bounds\n")
    except Exception as e:
        print("\nERROR: ", e)

# Step 1: Open file & parse line by line (x)
# Step 2: Test program to see what addresses should look like
# Step 3: Write sample file (x)
# Step 4: Add missing code & w/sample file
