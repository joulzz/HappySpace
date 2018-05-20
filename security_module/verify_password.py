from passlib.hash import sha256_crypt
import boto3
import os

def verify_password(password):
    s3 = boto3.resource("s3")
    s3.Bucket("smiledetection").download_file("password.txt", 'password.txt')
    with open("password.txt") as fp:
        target = fp.read()

    if sha256_crypt.verify(password, target):
        


