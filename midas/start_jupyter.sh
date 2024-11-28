#!/bin/bash
cls

DIR="$( cd "$( dirname "$0" )" && pwd )"

cd $DIR

date

echo Computador: "$( hostname )"        Usuario: "$( whoami )"

echo Aguarde enquanto iniciamos o notebook jupyter

jupyter notebook 

echo 'Pronto!'
pause