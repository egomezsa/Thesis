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


echo "Running partial only"
python PartialEnsamble.py GG
python PartialEnsamble.py GM
python PartialEnsamble.py GS
python PartialEnsamble.py SS
python PartialEnsamble.py SM
python PartialEnsamble.py MM
python PartialEnsamble.py MG
python PartialEnsamble.py MS

echo "Running series only"
python Series.py G
python Series.py M
