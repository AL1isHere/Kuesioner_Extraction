# app.py
# File utama untuk menampilkan UI (View) dan menjalankan aplikasi Streamlit.

import streamlit as st
import os
import model      # Mengimpor modul model
import controller # Mengimpor modul controller
import config     # Mengimpor modul config

# --- UI Streamlit ---
st.set_page_config(page_title="Ekstraksi Data Kuesioner", layout="wide")
st.title("📄 Aplikasi Ekstraksi Data dari Kuesioner PDF")
st.markdown("Aplikasi ini menggunakan AI untuk mendeteksi area pertanyaan dan mengenali karakter dari PDF kuesioner.")

# Muat model di awal aplikasi (akan di-cache)
# Pesan error akan ditampilkan di konsol/log jika gagal, dan di UI saat proses berjalan.
model_ssd_loaded_global = model.load_ssd_model(config.SSD_MODEL_PATH, config.SSD_NUM_CLASSES, config.DEVICE)
model_char_classifier_loaded_global = model.load_char_classifier_model(config.CHAR_CLASSIFIER_MODEL_PATH, config.CHAR_NUM_CLASSES, config.DEVICE)


# Inisialisasi session state
if 'current_page' not in st.session_state: st.session_state.current_page = 1
if 'all_results_data' not in st.session_state: st.session_state.all_results_data = []
if 'excel_buffer_data' not in st.session_state: st.session_state.excel_buffer_data = None
if 'processed_pdf_name' not in st.session_state: st.session_state.processed_pdf_name = ""
if 'processed_gender' not in st.session_state: st.session_state.processed_gender = ""


# --- Sidebar UI ---
with st.sidebar:
    st.header("⚙️ Pengaturan Input")
    uploaded_pdf_file_obj_ui = st.file_uploader("1. Unggah File PDF Kuesioner", type="pdf", key="pdf_uploader_widget_main")
    
    gender_options = ["pria", "perempuan"]
    selected_gender_ui_val = st.selectbox("2. Pilih Jenis Kelamin (untuk anotasi area)", gender_options, index=0, key="gender_selector_widget_main" )
    
    process_button_ui_val = st.button("🚀 Mulai Proses Ekstraksi", type="primary", disabled=(not uploaded_pdf_file_obj_ui))

# --- Logika Pemicu Controller ---
if process_button_ui_val and uploaded_pdf_file_obj_ui:
    if not model_ssd_loaded_global or not model_char_classifier_loaded_global:
        st.error("Model AI tidak berhasil dimuat. Proses tidak dapat dilanjutkan. Periksa pesan error di konsol atau di atas saat aplikasi pertama kali dimuat.")
    else:
        with st.container(): 
            st.info("Memulai proses ekstraksi... Mohon tunggu.")
            
            pdf_bytes_data_main = uploaded_pdf_file_obj_ui.getvalue() 
            
            # Panggil fungsi controller
            results_main, excel_buf_main = controller.run_extraction_workflow(
                pdf_bytes_data_main, 
                selected_gender_ui_val,
                model_ssd_loaded_global,
                model_char_classifier_loaded_global,
                config.DEVICE
            )
            
            # Simpan hasil ke session state untuk ditampilkan oleh View
            st.session_state.all_results_data = results_main
            st.session_state.excel_buffer_data = excel_buf_main
            st.session_state.current_page = 1 
            st.session_state.processed_pdf_name = uploaded_pdf_file_obj_ui.name
            st.session_state.processed_gender = selected_gender_ui_val
            
            if results_main:
                st.success("🎉 Proses ekstraksi selesai!")
            else:
                st.warning("Proses ekstraksi selesai, namun tidak ada hasil yang ditemukan atau terjadi error saat pemrosesan.")
            st.rerun() # Rerun untuk membersihkan spinner dan menampilkan bagian hasil


# --- Tampilan Hasil dan Paginasi ---
if st.session_state.all_results_data:
    st.markdown("---")
    st.subheader("📖 Hasil Ekstraksi")
    
    if st.session_state.excel_buffer_data:
        st.download_button(
            label="📥 Unduh Semua Hasil ke Excel",
            data=st.session_state.excel_buffer_data,
            file_name=f"hasil_ekstraksi_{st.session_state.processed_gender}_{st.session_state.processed_pdf_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_excel_button_top_view_main" 
        )
        st.markdown("---") 

    total_items_main = len(st.session_state.all_results_data)
    total_pages_main = (total_items_main + config.ITEMS_PER_PAGE - 1) // config.ITEMS_PER_PAGE

    if total_pages_main > 0:
        if st.session_state.current_page < 1: st.session_state.current_page = 1
        if st.session_state.current_page > total_pages_main: st.session_state.current_page = total_pages_main
        
        def display_pagination_controls(key_prefix_str):
            col_prev, col_page_info, col_next = st.columns([1, 3, 1])
            with col_prev:
                if st.button("⬅️ Sebelumnya", key=f"{key_prefix_str}_prev", disabled=(st.session_state.current_page <= 1)):
                    st.session_state.current_page -= 1
                    st.rerun() 
            with col_page_info:
                st.markdown(f"<p style='text-align: center; margin-top: 8px;'>Halaman {st.session_state.current_page} dari {total_pages_main}</p>", unsafe_allow_html=True)
            with col_next:
                if st.button("Berikutnya ➡️", key=f"{key_prefix_str}_next", disabled=(st.session_state.current_page >= total_pages_main)):
                    st.session_state.current_page += 1
                    st.rerun() 

        display_pagination_controls("top_pagination_main") 
        
        start_idx_main = (st.session_state.current_page - 1) * config.ITEMS_PER_PAGE
        end_idx_main = start_idx_main + config.ITEMS_PER_PAGE
        paginated_results_main = st.session_state.all_results_data[start_idx_main:end_idx_main]

        for res_idx_main, res_data_item_main in enumerate(paginated_results_main):
            st.markdown(f"**Data ke-{start_idx_main + res_idx_main + 1}**")
            col1_view_main, col2_view_main = st.columns([1,2]) 
            with col1_view_main:
                st.image(res_data_item_main["Image_PIL"], width=200, caption=f"Pertanyaan (ID): {res_data_item_main['ID_Pertanyaan']}") 
            with col2_view_main:
                st.write(f"**ID Pertanyaan:** {res_data_item_main['ID_Pertanyaan']}") 
                st.write(f"**Halaman:** {res_data_item_main['Halaman']}")
                st.write(f"**Teks Dikenali:** `{res_data_item_main['Teks']}`")
                st.write(f"**Avg. Confidence:** {res_data_item_main['Avg_Conf']}")
            st.divider()

        if total_pages_main > 1: 
            st.markdown("---") 
            display_pagination_controls("bottom_pagination_main") 
    else: 
        if st.session_state.get('processed_pdf_name'): 
             st.write("Tidak ada hasil ekstraksi untuk ditampilkan.")


elif process_button_ui_val and not uploaded_pdf_file_obj_ui: 
    st.error("Harap unggah file PDF terlebih dahulu.")


st.sidebar.markdown("---")
st.sidebar.info("Aplikasi ini dibuat untuk mendemonstrasikan ekstraksi data dari kuesioner menggunakan AI.")
