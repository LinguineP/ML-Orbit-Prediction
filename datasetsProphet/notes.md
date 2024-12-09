import problems

had to change in ssl.py
if trust is True or purpose.oid in trust:
certs.extend(cert)

~into:~

if trust is True or purpose.oid in trust: ##ssl cert not working celik
if "MUP Republike Srbije" not in str(cert):
certs.extend(cert)
