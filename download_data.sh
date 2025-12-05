#!/bin/bash

# Create directory
mkdir -p Data/TAU/Nikon_D810

# Download parts
echo "Downloading Part 1..."
curl -fOJ "https://download.fairdata.fi:443/download?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE3NjUxNDY0MzEsImRhdGFzZXQiOiJmMDU3MGEzZi0zZDc3LTRmNDQtOWVmMS05OWFiNDg3OGYxN2MiLCJmaWxlIjoiL05pa29uX0Q4MTBfaXNvdHJvcGljLnppcC4wMDEiLCJwcm9qZWN0IjoiMjAwMDQ2NCIsInJhbmRvbV9zYWx0IjoiMmFmZDY1ZTEifQ.ucXzCx9G047oYRpOp5ZvnlsIjUrYSL-lvTNHO1mIDU8"
echo "Downloading Part 2..."
curl -fOJ "https://download.fairdata.fi:443/download?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE3NjUxNDY1MjYsImRhdGFzZXQiOiJmMDU3MGEzZi0zZDc3LTRmNDQtOWVmMS05OWFiNDg3OGYxN2MiLCJmaWxlIjoiL05pa29uX0Q4MTBfaXNvdHJvcGljLnppcC4wMDIiLCJwcm9qZWN0IjoiMjAwMDQ2NCIsInJhbmRvbV9zYWx0IjoiMjU5MDUwMTYifQ.Bdoy_xjZlZlpnFs0rN26qO7SIrGIRdvf3wTXleKc8s4"
echo "Downloading Part 3..."
curl -fOJ "https://download.fairdata.fi:443/download?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE3NjUxNDY1NDAsImRhdGFzZXQiOiJmMDU3MGEzZi0zZDc3LTRmNDQtOWVmMS05OWFiNDg3OGYxN2MiLCJmaWxlIjoiL05pa29uX0Q4MTAuemlwLjAwMyIsInByb2plY3QiOiIyMDAwNDY0IiwicmFuZG9tX3NhbHQiOiJjNTEzMjM0NiJ9.w7YM2jXVED4Y2kYUYpBY1BjKDew1lVcawOsSzE0nwEw"
echo "Downloading Part 4..."
curl -fOJ "https://download.fairdata.fi:443/download?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE3NjUxNDY1NTYsImRhdGFzZXQiOiJmMDU3MGEzZi0zZDc3LTRmNDQtOWVmMS05OWFiNDg3OGYxN2MiLCJmaWxlIjoiL05pa29uX0Q4MTBfaXNvdHJvcGljLnppcC4wMDQiLCJwcm9qZWN0IjoiMjAwMDQ2NCIsInJhbmRvbV9zYWx0IjoiMTZkOTc5ZmYifQ.VmUnmTGXgJCDtpXxe1dON_aRL5jKzqhIqJJ_DiBg-3Y"
# Combine parts
echo "Combining parts..."
cat Nikon_D810_isotropic.zip.001 Nikon_D810_isotropic.zip.002 Nikon_D810_isotropic.zip.003 Nikon_D810_isotropic.zip.004 > Nikon_D810_combined.zip

# Extract
echo "Extracting to Data/Nikon_D810..."
unzip -q Nikon_D810_combined.zip -d Data/Nikon_D810

# Cleanup
echo "Cleaning up zip files..."
rm Nikon_D810_isotropic.zip.* Nikon_D810_combined.zip

echo "Done! Dataset is in Data/Nikon_D810"
