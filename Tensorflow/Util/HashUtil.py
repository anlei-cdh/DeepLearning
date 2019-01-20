# -*- coding: utf-8 -*-
import hashlib

class HashUtil():

    hash_256 = hashlib.sha256()

    def getHash(self,hash_str):
        self.hash_256.update(hash_str.encode('utf-8'))
        hash_256_value = self.hash_256.hexdigest()
        return int(hash_256_value, 16) % (2**63) #9223372036854775808