# controller.py
# Mengelola alur aplikasi dan interaksi antara Model dan View.

import streamlit as st
import os
import json
import numpy as np
import model
import config # Import file config

def run_extraction_workflow(uploaded_pdf_bytes, selected_gender_str, 
                            model_s, model_c, device_obj):
    """
    Fungsi utama untuk mengontrol proses ekstraksi.
    Mengambil input, memanggil fungsi model, dan mengembalikan hasil.
    """
    all_extracted_results = []
    
    # Dapatkan transformasi dari model
    transform_s = model.get_ssd_transform()
    transform_c = model.get_char_transform()
    
    # 1. Load data anotasi area
    annotation_crop_filename = f"anotasi_{selected_gender_str}.json"
    annotation_crop_path = os.path.join(config.APP_DATA_PATH, annotation_crop_filename)

    if not os.path.exists(annotation_crop_path):
        st.error(f"File anotasi area '{annotation_crop_filename}' tidak ditemukan. Pastikan ada di: '{config.APP_DATA_PATH}'")
        return [], None
            
    with open(annotation_crop_path, "r") as f:
        crop_area_data = json.load(f)

    # 2. Konversi PDF ke gambar
    pdf_images_pil = model.model_convert_pdf(uploaded_pdf_bytes) 
    if not pdf_images_pil:
        st.error("Gagal mengkonversi PDF ke gambar.")
        return [], None

    # Inisialisasi progress bar Streamlit
    progress_text_area = "Memproses anotasi pertanyaan PDF..."
    my_bar = st.progress(0.0, text=f"{progress_text_area} (0/{len(crop_area_data)})")
    num_tasks_total = len(crop_area_data)

    # 3. Iterasi dan proses setiap area anotasi
    for task_idx, task in enumerate(crop_area_data): 
        my_bar.progress((task_idx + 1) / num_tasks_total, text=f"{progress_text_area} ({task_idx+1}/{num_tasks_total})")

        annotations_result = task['annotations'][0]['result']
        
        for item_idx, item in enumerate(annotations_result):
            if item['type'] == 'rectanglelabels' and item['from_name'] == 'box':
                box_id = item['id']
                field_annotation_value = item['value']
                original_w = item['original_width']
                original_h = item['original_height']
                
                id_pertanyaan = "N/A" 
                try:
                    id_pertanyaan = next(r['value']['text'][0] for r in annotations_result if r['from_name'] == 'nomor' and r['id'] == box_id)
                except StopIteration: pass

                halaman_str = "0"
                try:
                    halaman_str = next(r['value']['text'][0] for r in annotations_result if r['from_name'] == 'halaman' and r['id'] == box_id)
                    page_idx = int(halaman_str) - 1
                    if not (0 <= page_idx < len(pdf_images_pil)): continue
                    current_page_image = pdf_images_pil[page_idx]
                except (StopIteration, ValueError): continue
                
                # Panggil model untuk crop area pertanyaan
                field_image_pil = model.model_crop_region(current_page_image, field_annotation_value, original_w, original_h)
                
                # Panggil model untuk deteksi karakter
                char_boxes_detected = model.model_detect_chars(
                    field_image_pil, model_s, device_obj, transform_s
                )
                
                recognized_string = ""
                confidences = []
                for _, char_box_coords in enumerate(char_boxes_detected):
                    x1, y1, x2, y2 = char_box_coords
                    if x1 >= x2 or y1 >= y2: continue
                    
                    char_image_pil_single = field_image_pil.crop((x1, y1, x2, y2)) 
                    if char_image_pil_single.width == 0 or char_image_pil_single.height == 0: continue
                    
                    # Panggil model untuk klasifikasi karakter
                    char_pred, confidence = model.model_classify_char(
                        char_image_pil_single, model_c, device_obj, transform_c
                    )
                    
                    if char_pred == "?" or confidence < config.CHAR_CLASSIFICATION_THRESHOLD : 
                        recognized_string += "?" 
                    else:
                        recognized_string += char_pred
                    confidences.append(confidence)

                avg_confidence = np.mean(confidences) if confidences else 0.0
                all_extracted_results.append({ 
                    "ID_Pertanyaan": id_pertanyaan, "Halaman": halaman_str, 
                    "Teks": recognized_string, "Avg_Conf": f"{avg_confidence:.2f}",
                    "Image_PIL": field_image_pil, 
                })
    
    my_bar.empty() 

    # 4. Panggil model untuk membuat laporan Excel
    excel_buffer_result = model.create_excel_report(all_extracted_results)
    return all_extracted_results, excel_buffer_result
