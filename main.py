from PIL import Image
import numpy
import math

approximation_ratio = float(input("Введите процент аппроксимации: "))
img_path = input("\033[0mВведите путь к файлу: ")
print()

rgba_dict = {
    0: "\033[31mRed-channel\033[0m",
    1: "\033[32mGreen-channel\033[0m",
    2: "\033[34mBlue-channel\033[0m"
}

image = Image.open(img_path)
matrix = numpy.array(image)

channels = []
for i in range(matrix.shape[-1]):  # R, G, B
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
    approximation_coefficient = math.ceil((100 - approximation_ratio) / 100 * Sigma.shape[0])
    print(f"Approximation coefficient: \033[33m{approximation_coefficient}\033[0m")

    approx_U = U[:, :approximation_coefficient]
    approx_Sigma = numpy.diag(Sigma[:approximation_coefficient])
    approx_V_T = V_T[:approximation_coefficient, :]

    approx_A = approx_U @ approx_Sigma @ approx_V_T
    approx_A = numpy.clip(approx_A, 0, 255)
    approx_A = numpy.round(approx_A)

    channels.append(approx_A)

image_name = "approx_image.jpg"
print("Creation approx image...")

approx_matrix = numpy.stack(channels, axis=2).astype(numpy.uint8)
approx_image = Image.fromarray(approx_matrix, mode="RGB")

approx_image.save(image_name)
print("\033[42mDone!\033[0m")