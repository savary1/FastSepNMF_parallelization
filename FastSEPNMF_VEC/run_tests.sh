for i in {1..10}
do
    echo "/n/n ############# Ejecución número $i #############"
    echo "/n/n ############# Ejecución número $i #############" >> results.txt
    ./FastSEPNMF ../../Images/Grande.bsq ../../Images/Grande.hdr 30 1 >> results.txt
    echo "Fin"
done