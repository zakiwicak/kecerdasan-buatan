#input value data 
nilai_ulangan_harian = 75
nilai_ulangan_tengah_semester = 80
nilai_ulangan_akhir_semester = 90
kehadiran_siswa = 0.8  # karena 80% dibuat desimal

# Bobot dan Bias random
w11 = 0.5
w21 = 0.6
w31 = 0.7
w41 = 0.8
b1 = 0
w12 = 0.9
w22 = 1.0
w32 = 1.1
w42 = 1.2
b2 = 0

# cari feedforward
z1 = (w11 * nilai_ulangan_harian) + (w21 * nilai_ulangan_tengah_semester) + (w31 * nilai_ulangan_akhir_semester) + (w41 * kehadiran_siswa) + b1
a1 = 1 / (1 + 2.71828**(-z1))

z2 = (w12 * nilai_ulangan_harian) + (w22 * nilai_ulangan_tengah_semester) + (w32 * nilai_ulangan_akhir_semester) + (w42 * kehadiran_siswa) + b2
a2 = 1 / (1 + 2.71828**(-z2))

z_output = (a1 * w11) + (a2 * w12)
output = 1 / (1 + 2.71828**(-z_output))

# cari feedback
target_output = 1  # Misalkan target kelulusan adalah 1 (lulus, karena kasus saya mencari lulus\tdk lulus)
error = target_output - output

# rums turunan fungsi sigmoid
sigmoid_prime = output * (1 - output)

# gradien pada output layer
delta_output = error * sigmoid_prime

# gradien pada hidden layer
delta_h1 = (delta_output * w11) * (a1 * (1 - a1))
delta_h2 = (delta_output * w12) * (a2 * (1 - a2))

# menentukan learning rate
alpha = 0.1

# update pada bobot dan bias berdasarkan hitungan
w11 = w11 + alpha * delta_h1 * nilai_ulangan_harian
w21 = w21 + alpha * delta_h1 * nilai_ulangan_tengah_semester
w31 = w31 + alpha * delta_h1 * nilai_ulangan_akhir_semester
w41 = w41 + alpha * delta_h1 * kehadiran_siswa
b1 = b1 + alpha * delta_h1

w12 = w12 + alpha * delta_h2 * nilai_ulangan_harian
w22 = w22 + alpha * delta_h2 * nilai_ulangan_tengah_semester
w32 = w32 + alpha * delta_h2 * nilai_ulangan_akhir_semester
w42 = w42 + alpha * delta_h2 * kehadiran_siswa
b2 = b2 + alpha * delta_h2

print("Bobot dan Bias Baru:")
print("w11:", w11, "w21:", w21, "w31:", w31, "w41:", w41, "b1:", b1)
print("w12:", w12, "w22:", w22, "w32:", w32, "w42:", w42, "b2:", b2)

if output >= 0.5:
    print("\nkemungkinan lulus")
else:
    print("\nkemungkinan tidak lulus.")