#!/bin/bash

echo "Exhaustive: Integers"
for (( n = 32; n <= 352; n+=32 )); do
  for (( i = 3; i <= 8; i++ )); do
    out3d="images/healpix_int${i}_nside${n}_exhaustive.html"
    if [ ! -f "$out3d" ]; then
      python3 ./vector_distribution_analysis.py  \
        --format "int$i" \
        --mode exhaustive  \
        --healpix  \
        --nside "$n"  \
        --out3d "$out3d"
    else
      echo "already created $out3d"
    fi
  done
done

echo "Exhaustive: e_m_"

for (( n = 32; n <= 352; n+=32 )); do
  for (( e = 1; e <= 5; e++ )); do
    for (( m = 2; m <= 10; m++ )); do
    out3d="images/healpix_e${e}m${m}_nside${n}_exhaustive.html"
    if [ ! -f "$out3d" ]; then
      python3 ./vector_distribution_analysis.py  \
        --format fp16 -e "$e" -m "$m" \
        --mode exhaustive  \
        --healpix  \
        --nside "$n"  \
        --out3d "$out3d"
    else
      echo "already created $out3d"
    fi

    done
  done
done

echo "integer with gaussian and standard deviations"
for (( n = 64; n <= 320; n+=64 )); do
  for (( i = 3; i <= 8; i++ )); do
    for (( std = 1; std <= 11; std+=2 )); do
      out3d="images/healpix_int${i}_gaussian_std${std}_1000000_healpix_${n}.html"
      if [ ! -f "$out3d" ]; then
        echo "Gaussian: int${i}"
        python3 ./vector_distribution_analysis.py  \
          --format "int${i}" \
          --mode gaussian  \
          --num 1000000  \
          --std "$std" \
          --healpix  \
          --nside "$n"  \
          --out3d "$out3d"
      else
        echo "already created $out3d"
      fi
    done
  done
done

echo "e_m_ with decimal standard deviation"
for (( n = 64; n <= 320; n+=64 )); do
  for (( e = 1; e <= 5; e++ )); do
    for (( m = 2; m <= 10; m++ )); do
      for (( std = 1; std <= 9; std+=2 )); do
        out3d="images/healpix_e${e}m${m}_gaussian_1000000_healpix_${n}_std0.${std}.html"
        if [ ! -f "$out3d" ]; then
          echo "Gaussian: e${e}m${m}"
          python3 ./vector_distribution_analysis.py  \
            --format fp16 -e "$e" -m "$m"  \
            --mode gaussian  \
            --num 1000000  \
            --healpix  \
            --std "0.$std" \
            --nside "$n"  \
            --out3d "$out3d"
        else
          echo "already created $out3d"
        fi
      done
    done
  done
done

echo "e_m_ with integer standard deviation"
for (( n = 64; n <= 320; n+=64 )); do
  for (( e = 1; e <= 5; e++ )); do
    for (( m = 2; m <= 10; m++ )); do
      for (( std = 1; std <= 9; std+=2 )); do
        out3d="images/healpix_e${e}m${m}_gaussian_1000000_healpix_${n}_std${std}.html"
        if [ ! -f "$out3d" ]; then
          echo "Gaussian: e${e}m${m}"
          python3 ./vector_distribution_analysis.py  \
            --format fp16 -e "$e" -m "$m"  \
            --mode gaussian  \
            --num 1000000  \
            --healpix  \
            --std "$std" \
            --nside "$n"  \
            --out3d "$out3d"
        else
          echo "already created $out3d"
        fi
      done
    done
  done
done

