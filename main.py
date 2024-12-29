from PIL import Image
import numpy
import math

rgba_dict = {
    0: "Red-channel",
    1: "Green-channel",
    2: "Blue-channel",
    3: "Alpha-channel"
}

image = Image.open('img.jpg')
matrix = numpy.array(image)

compression_ratio = float(input("Введите процент сжатия: "))

channels = []
for i in range(matrix.shape[2]):  # R, G, B[, A]
    print(f"Approximation {rgba_dict[i]} matrix...")
    U, Sigma, V_T = numpy.linalg.svd(matrix[:, :, i], full_matrices=False)

    # Значеня для усечения нулевых строк и столбцов
    zero_count_Sigma = numpy.count_nonzero(Sigma == 0)
    column_count_U = numpy.shape(U)[1] - zero_count_Sigma
    row_count_V_T = numpy.shape(V_T)[0] - zero_count_Sigma

    # Усечение нулевых строк и столбцов
    if zero_count_Sigma > 0:
        U = U[:, :column_count_U]
        if zero_count_Sigma == Sigma.shape[0]:
            Sigma = Sigma[:0]
        else:
            Sigma = Sigma[:zero_count_Sigma - 1]
        V_T = V_T[:row_count_V_T, :]

    # Аппроксимизация матриц
    approximation_coefficient = math.ceil((100 - compression_ratio) / 100 * Sigma.shape[0])
    print("Approximation coefficient:", approximation_coefficient)

    approx_U = U[:, :approximation_coefficient]
    approx_Sigma = numpy.diag(Sigma[:approximation_coefficient])
    approx_V_t = V_T[:approximation_coefficient, :]

    approx_A = approx_U @ approx_Sigma @ approx_V_t
    approx_A = numpy.round(approx_A)
    channels.append(approx_A.astype('uint8'))

approx_matrix = numpy.stack(channels, axis=2).astype(numpy.uint8)

print("Creation approx image...")
image_name = "approx_image."
if matrix.shape[-1] == 3:
    image_name += "jpg"
    approx_image = Image.fromarray(approx_matrix, mode="RGB")
elif matrix.shape[-1] == 4:
    image_name += "png"
    approx_image = Image.fromarray(approx_matrix, mode="RGBA")
approx_image.save(image_name)
print("Done!")