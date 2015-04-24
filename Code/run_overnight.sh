#!/bin/bash


echo "Running audio only"
python audioNB.py S
python audioNB.py G
python audioNB.py M


echo "Running bow only"
python classify_BOW.py S
python classify_BOW.py G
python classify_BOW.py M


echo "Running ensemble only"
python Ensamble.py GGG
python Ensamble.py GMG
python Ensamble.py GGM
python Ensamble.py GMM
python Ensamble.py GSG
python Ensamble.py GSM
python Ensamble.py GMS
python Ensamble.py GSS
python Ensamble.py GGS



echo "Running partial audio only"
python PartialEnsamble.py GG1
python PartialEnsamble.py GM1
python PartialEnsamble.py GS1
python PartialEnsamble.py MM1
python PartialEnsamble.py MG1
python PartialEnsamble.py MS1
python PartialEnsamble.py SM1
python PartialEnsamble.py SG1
python PartialEnsamble.py SS1

echo "Running partial lyrics only"
python PartialEnsamble.py GG2
python PartialEnsamble.py GM2
python PartialEnsamble.py GS2
python PartialEnsamble.py MM2
python PartialEnsamble.py MG2
python PartialEnsamble.py MS2
python PartialEnsamble.py SM2
python PartialEnsamble.py SG2
python PartialEnsamble.py SS2

echo "Running series only"
python Series.py G
python Series.py M
# python Series.py S
