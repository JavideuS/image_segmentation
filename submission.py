from PIL import Image
import os
import numpy as np # librería para hacer calculos de álgebra lineal
import pandas as pd # tratamiento de datos, archivo CSV  I/O (ej. pd.read_csv)
import zlib
import base64
import typing as t
import cv2
from pycocotools import _mask as coco_mask


def output_save(output, save_directory, original_dir, verbose=False):
    batch_size = output.shape[0]
    num_masks = output.shape[1]
    image_names = os.listdir(original_dir)

    for d in image_names:
        subfolder = save_directory + '/' + d.split('.')[0]
        if not os.path.exists(subfolder):
            os.mkdir(subfolder)

    dirs = os.listdir(save_directory)
    print(dirs)

    for i in range(batch_size):
        for j in range(num_masks):
            # Get the mask from the tensor and convert to a binary mask (0 or 1)
            mask = output[i, j]
            binary_mask = (mask > 0.01).int()  # Example threshold to create binary mask

            # Convert to numpy array
            binary_mask_np = binary_mask.numpy().astype(np.uint8) * 255  # Convert to 0 and 255

            # Convert to PIL image
            img = Image.fromarray(binary_mask_np)

            # Save the image
            img_path = save_directory + '/' + dirs[i] + '/' + f'batch_{i}_mask_{j}_Obj.png'
            img.save(img_path)
            if verbose:
                print(f"Saved {img_path}")
    print(f'All images saved on {save_directory}')


def encode_binary_mask(mask: np.ndarray) -> t.Text:
    """Convierte una máscara binaria a texto ascii de codificación de desafío OID."""
    # Comprobar máscara de entrada
    if mask.dtype != np.uint8:
        raise ValueError(
            "encode_binary_mask espera una máscara binaria, dtype recibido == %s" % mask.dtype
        )

    mask = np.squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError(
            "encode_binary_mask espera una máscara 2d, forma recibida == %s" % mask.shape
        )

    # Convertir la máscara de entrada en la entrada prevista de COCO API
    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)

    # codoficar máscara RLE
    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

    # Compresión y codificación base64
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)
    return base64_str.decode("utf-8")  # Devolver como cadena


def encode_to_csv(saved_dir, verbose=False):
    encoded_masks_dict = {}

    # Recorrer todas las carpetas del directorio de entrada
    for root, dirs, files in os.walk(saved_dir):
        try:
            dirs.remove('all')
            print(f"Carpeta 'all'eliminada en {root}")
        except:
            print(f"Carpeta 'all' no encontrada en {root}")
        for dir_name in dirs:
            if verbose:
                print(f"Procesando carpeta {dir_name} ...")
            folder_path = os.path.join(root, dir_name)
            # Inicializar las listas para almacenar las máscaras de cada carpeta
            encoded_masks_str = ""
            binary_masks_list = []
            first_image_path = None
            width = None
            height = None
            # Recorrer todos los archivos de la carpeta actual
            for filename in os.listdir(folder_path):
                if ((filename.lower().endswith('.png')) and ("Obj" in filename)):
                    # Leer la imagen
                    image_path = os.path.join(folder_path, filename)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Lectura en escala de grises
                    # Obtener anchura y altura de la imagen
                    height, width = image.shape

                    # Convertir a máscara binaria  (thresholding)
                    _, binary_mask = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

                    # Encode the binary mask
                    encoded_mask = encode_binary_mask(binary_mask)

                    # Codificar la máscara binaria
                    if encoded_masks_str:
                        encoded_masks_str += " " + encoded_mask
                    else:
                        encoded_masks_str = encoded_mask

                    # Añade la máscara codificada y la máscara binaria a las listas
                    binary_masks_list.append(binary_mask)
            print(f"La codificación de la carpeta {dir_name} se ha realizado correctamente!")
            # Añade el nombre de la carpeta como ID de la imagen y su información asociada al diccionario
            encoded_masks_dict[dir_name] = {
                'Width': width,  # Ancho
                'Height': height,  # Alto
                'EncodedMasks': encoded_masks_str,  # máscaras codificadas
                'Masks': binary_masks_list  # lista de máscaras binarias
            }
            print("Codificación finalizada!")
    return encoded_masks_dict


def final_encode(encoded_masks_dict, name):
    # Crear un DataFrame para exportación CSV (excluyendo el 'DecodedMask', máscaras decodificadas)
    df = pd.DataFrame.from_dict(encoded_masks_dict, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'ID'}, inplace=True)

    # Eliminar la columna "DecodedMask" del DataFrame
    df.drop(columns=['Masks'], inplace=True)

    # Guardar el DataFrame en un archivo CSV
    save_path = "./" + name + "_COCO_COMPRESSED.csv"
    df.to_csv(save_path, index=False)

    return save_path


def main(output, save_directory, original_dir , name):
    output_save(output, save_directory, original_dir)
    encoded_masks_dict = encode_to_csv(save_directory)
    save_path = final_encode(encoded_masks_dict , name)
    return save_path

