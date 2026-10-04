from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn

from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from torchvision import transforms
from transformers import ViTConfig, ViTForImageClassification




MODEL_REPO_ID = (
    "Rahat171/"
    "BD_Handwritten_Distrcit_Upazila_Recognition"
)

MODEL_REVISION = "main"


MODEL_VERSION = "v2"


MODEL_FILES = {
    "v1": {
        "vit": "best_vit_base_custom_mlp_v1.pth",
        "convnext": "best_convnext_tiny_custom_mlp_v1.pth",
    },
    "v2": {
        "vit": "best_vit_base_custom_mlp_v2.pth",
        "convnext": "best_convnext_tiny_custom_mlp_v2.pth",
    },
}


if MODEL_VERSION not in MODEL_FILES:
    raise ValueError(
        f"Unknown model version: {MODEL_VERSION}"
    )


VIT_MODEL_FILENAME = (
    MODEL_FILES[MODEL_VERSION]["vit"]
)

CONVNEXT_MODEL_FILENAME = (
    MODEL_FILES[MODEL_VERSION]["convnext"]
)



NUM_CLASSES = 496
IMAGE_SIZE = 224
CONVNEXT_VARIANT = "tiny"
USE_CROP = False



NORM_MEAN = [
    0.485,
    0.456,
    0.406,
]

NORM_STD = [
    0.229,
    0.224,
    0.225,
]


VIT_WEIGHT = 0.15
CONVNEXT_WEIGHT = 0.85

VOTING_WEIGHTS = [
    VIT_WEIGHT,
    CONVNEXT_WEIGHT,
]




CLASS_NAMES = [
    "Abhaynagar Upazila,Jashore",
    "Adamdighi Upazila,Bogura",
    "Aditmari Upazila,Lalmonirhat",
    "Agailjhara Upazila,Barishal",
    "Ajmiriganj Upazila,Habiganj",
    "Akhaura Upazila,Brahmanbaria",
    "Akkelpur Upazila,Joypurhat",
    "Alamdanga Upazila,Chuadanga",
    "Alfadanga Upazila,Faridpur",
    "Ali Kadam Upazila,Bandarban",
    "Amtali Upazila,Barguna",
    "Anwara Upazila,Chattogram",
    "Araihazar Upazila,Narayanganj",
    "Ashuganj Upazila,Brahmanbaria",
    "Assasuni Upazila,Satkhira",
    "Atgharia Upazila,Pabna",
    "Atpara Upazila,Netrokona",
    "Atrai Upazila,Naogaon",
    "Atwari Upazila,Panchagarh",
    "Austagram Upazila,Kishoreganj",
    "Babuganj Upazila,Barishal",
    "Badalgachhi Upazila,Naogaon",
    "Badarganj Upazila,Rangpur",
    "Bagaichhari Upazila,Rangamati",
    "Bagatipara Upazila,Natore",
    "Bagerhat Sadar Upazila,Bagerhat",
    "Bagha Upazila,Rajshahi",
    "Bagherpara Upazila,Jashore",
    "Bagmara Upazila,Rajshahi",
    "Bahubal Upazila,Habiganj",
    "Bajitpur Upazila,Kishoreganj",
    "Bakerganj Upazila,Barishal",
    "Baksiganj Upazila,Jamalpur",
    "Balaganj Upazila,Sylhet",
    "Baliadangi Upazila,Thakurgaon",
    "Baliakandi Upazila,Rajbari",
    "Bamna Upazila,Barguna",
    "Banaripara Upazila,Barishal",
    "Bancharampur Upazila,Brahmanbaria",
    "Bandar Upazila,Narayanganj",
    "Bandarban Sadar Upazila,Bandarban",
    "Baniyachong Upazila,Habiganj",
    "Banshkhali Upazila,Chattogram",
    "Baraigram Upazila,Natore",
    "Barguna Sadar Upazila,Barguna",
    "Barhatta Upazila,Netrokona",
    "Barishal Sadar Upazila,Barishal",
    "Barkal Upazila,Rangamati",
    "Barlekha Upazila,Moulvibazar",
    "Barura Upazila,Cumilla",
    "Basail Upazila,Tangail",
    "Batiaghata Upazila,Khulna",
    "Bauphal Upazila,Patuakhali",
    "Beanibazar Upazila,Sylhet",
    "Begumganj Upazila,Noakhali",
    "Belabo Upazila,Narsingdi",
    "Belaichhari Upazila,Rangamati",
    "Belkuchi Upazila,Sirajganj",
    "Bera Upazila,Pabna",
    "Betagi Upazila,Barguna",
    "Bhairab Upazila,Kishoreganj",
    "Bhaluka Upazila,Mymensingh",
    "Bhandaria Upazila,Pirojpur",
    "Bhanga Upazila,Faridpur",
    "Bhangura Upazila,Pabna",
    "Bhedarganj Upazila,Shariatpur",
    "Bheramara Upazila,Kushtia",
    "Bhola Sadar Upazila,Bhola",
    "Bholahat Upazila,Chapai Nawabganj",
    "Bhuapur Upazila,Tangail",
    "Bhurungamari Upazila,Kurigram",
    "Bijoynagar Upazila,Brahmanbaria",
    "Biral Upazila,Dinajpur",
    "Birampur Upazila,Dinajpur",
    "Birganj Upazila,Dinajpur",
    "Bishwamvarpur Upazila,Sunamganj",
    "Bishwanath Upazila,Sylhet",
    "Boalkhali Upazila,Chattogram",
    "Boalmari Upazila,Faridpur",
    "Bochaganj Upazila,Dinajpur",
    "Boda Upazila,Panchagarh",
    "Bogura Sadar Upazila,Bogura",
    "Brahmanbaria Sadar Upazila,Brahmanbaria",
    "Brahmanpara Upazila,Cumilla",
    "Burhanuddin Upazila,Bhola",
    "Burichang Upazila,Cumilla",
    "Chagalnaiya Upazila,Feni",
    "Chakaria Upazila,Cox_s Bazar",
    "Chandanaish Upazila,Chattogram",
    "Chandina Upazila,Cumilla",
    "Chandpur Sadar Upazila,Chandpur",
    "Char Fasson Upazila,Bhola",
    "Char Rajibpur Upazila,Kurigram",
    "Charbhadrasan Upazila,Faridpur",
    "Charghat Upazila,Rajshahi",
    "Chatkhil Upazila,Noakhali",
    "Chatmohar Upazila,Pabna",
    "Chauddagram Upazila,Cumilla",
    "Chaugachha Upazila,Jashore",
    "Chauhali Upazila,Sirajganj",
    "Chhatak Upazila,Sunamganj",
    "Chilmari Upazila,Kurigram",
    "Chirirbandar Upazila,Dinajpur",
    "Chitalmari Upazila,Bagerhat",
    "Chuadanga Sadar Upazila,Chuadanga",
    "Chunarughat Upazila,Habiganj",
    "Companiganj Upazila,Noakhali",
    "Companiganj Upazila,Sylhet",
    "Cox_s Bazar Sadar Upazila,Cox_s Bazar",
    "Cumilla Sadar Dakshin Upazila,Cumilla",
    "Cumilla Sadar Upazila,Cumilla",
    "Dacope Upazila,Khulna",
    "Daganbhuiyan Upazila,Feni",
    "Dakshin Surma Upazila,Sylhet",
    "Damudya Upazila,Shariatpur",
    "Damurhuda Upazila,Chuadanga",
    "Dasar Upazila,Madaripur",
    "Dashmina Upazila,Patuakhali",
    "Daudkandi Upazila,Cumilla",
    "Daulatkhan Upazila,Bhola",
    "Daulatpur Upazila,Kushtia",
    "Daulatpur Upazila,Manikganj",
    "Debhata Upazila,Satkhira",
    "Debidwar Upazila,Cumilla",
    "Debiganj Upazila,Panchagarh",
    "Delduar Upazila,Tangail",
    "Derai Upazila,Sunamganj",
    "Dewanganj Upazila,Jamalpur",
    "Dhamoirhat Upazila,Naogaon",
    "Dhamrai Upazila,Dhaka",
    "Dhanbari Upazila,Tangail",
    "Dharamapasha Upazila,Sunamganj",
    "Dhobaura Upazila,Mymensingh",
    "Dhunot Upazila,Bogura",
    "Dhupchanchia Upazila,Bogura",
    "Dighalia Upazila,Khulna",
    "Dighinala Upazila,Khagrachhari",
    "Dimla Upazila,Nilphamari",
    "Dinajpur Sadar Upazila,Dinajpur",
    "Dohar Upazila,Dhaka",
    "Domar Upazila,Nilphamari",
    "Dowarabazar Upazila,Sunamganj",
    "Dumki Upazila,Patuakhali",
    "Dumuria Upazila,Khulna",
    "Durgapur Upazila,Netrokona",
    "Durgapur Upazila,Rajshahi",
    "Eidgaon Upazila,Cox_s Bazar",
    "Fakirhat Upazila,Bagerhat",
    "Faridganj Upazila,Chandpur",
    "Faridpur Sadar Upazila,Faridpur",
    "Faridpur Upazila,Pabna",
    "Fatikchhari Upazila,Chattogram",
    "Fenchuganj Upazila,Sylhet",
    "Feni Sadar Upazila,Feni",
    "Fulbaria Upazila,Mymensingh",
    "Fulgazi Upazila,Feni",
    "Gabtali Upazila,Bogura",
    "Gafargaon Upazila,Mymensingh",
    "Gaibandha Sadar Upazila,Gaibandha",
    "Galachipa Upazila,Patuakhali",
    "Gangachhara Upazila,Rangpur",
    "Gangni Upazila,Meherpur",
    "Gauripur Upazila,Mymensingh",
    "Gaurnadi Upazila,Barishal",
    "Gazaria Upazila,Munshiganj",
    "Gazipur Sadar Upazila,Gazipur",
    "Ghatail Upazila,Tangail",
    "Ghior Upazila,Manikganj",
    "Ghoraghat Upazila,Dinajpur",
    "Goalandaghat Upazila,Rajbari",
    "Gobindaganj Upazila,Gaibandha",
    "Godagari Upazila,Rajshahi",
    "Golapganj Upazila,Sylhet",
    "Gomastapur Upazila,Chapai Nawabganj",
    "Gopalganj Sadar Upazila,Gopalganj",
    "Gopalpur Upazila,Tangail",
    "Gosairhat Upazila,Shariatpur",
    "Gowainghat Upazila,Sylhet",
    "Guimara Upazila,Khagrachhari",
    "Gurudaspur Upazila,Natore",
    "Habiganj Sadar Upazila,Habiganj",
    "Haimchar Upazila,Chandpur",
    "Hakimpur Upazila,Dinajpur",
    "Haluaghat Upazila,Mymensingh",
    "Harinakunda Upazila,Jhenaidah",
    "Haripur Upazila,Thakurgaon",
    "Harirampur Upazila,Manikganj",
    "Hathazari Upazila,Chattogram",
    "Hatibandha Upazila,Lalmonirhat",
    "Hatiya Upazila,Noakhali",
    "Haziganj Upazila,Chandpur",
    "Hizla Upazila,Barishal",
    "Homna Upazila,Cumilla",
    "Hossainpur Upazila,Kishoreganj",
    "Indurkani Upazila,Pirojpur",
    "Ishwardi Upazila,Pabna",
    "Ishwarganj Upazila,Mymensingh",
    "Islampur Upazila,Jamalpur",
    "Itna Upazila,Kishoreganj",
    "Jagannathpur Upazila,Sunamganj",
    "Jaintiapur Upazila,Sylhet",
    "Jaldhaka Upazila,Nilphamari",
    "Jamalganj Upazila,Sunamganj",
    "Jamalpur Sadar Upazila,Jamalpur",
    "Jashore Sadar Upazila,Jashore",
    "Jhalokati Sadar Upazila,Jhalokati",
    "Jhenaidah Sadar Upazila,Jhenaidah",
    "Jhenaigati Upazila,Sherpur",
    "Jhikargachha Upazila,Jashore",
    "Jibannagar Upazila,Chuadanga",
    "Joypurhat Sadar Upazila,Joypurhat",
    "Juraichhari Upazila,Rangamati",
    "Juri Upazila,Moulvibazar",
    "Kabirhat Upazila,Noakhali",
    "Kachua Upazila,Bagerhat",
    "Kachua Upazila,Chandpur",
    "Kahaloo Upazila,Bogura",
    "Kaharole Upazila,Dinajpur",
    "Kalai Upazila,Joypurhat",
    "Kalapara Upazila,Patuakhali",
    "Kalaroa Upazila,Satkhira",
    "Kalia Upazila,Narail",
    "Kaliakair Upazila,Gazipur",
    "Kaliganj Upazila,Gazipur",
    "Kaliganj Upazila,Jhenaidah",
    "Kaliganj Upazila,Lalmonirhat",
    "Kaliganj Upazila,Satkhira",
    "Kalihati Upazila,Tangail",
    "Kalkini Upazila,Madaripur",
    "Kalmakanda Upazila,Netrokona",
    "Kalukhali Upazila,Rajbari",
    "Kamalganj Upazila,Moulvibazar",
    "Kamalnagar Upazila,Lakshmipur",
    "Kamarkhanda Upazila,Sirajganj",
    "Kanaighat Upazila,Sylhet",
    "Kapasia Upazila,Gazipur",
    "Kaptai Upazila,Rangamati",
    "Karimganj Upazila,Kishoreganj",
    "Karnaphuli Upazila,Chattogram",
    "Kasba Upazila,Brahmanbaria",
    "Kashiani Upazila,Gopalganj",
    "Kathalia Upazila,Jhalokati",
    "Katiadi Upazila,Kishoreganj",
    "Kaunia Upazila,Rangpur",
    "Kawkhali (Betbunia) Upazila,Rangamati",
    "Kawkhali Upazila,Pirojpur",
    "Kazipur Upazila,Sirajganj",
    "Kendua Upazila,Netrokona",
    "Keraniganj Upazila,Dhaka",
    "Keshabpur Upazila,Jashore",
    "Khagrachhari Sadar Upazila,Khagrachhari",
    "Khaliajuri Upazila,Netrokona",
    "Khansama Upazila,Dinajpur",
    "Khetlal Upazila,Joypurhat",
    "Khoksa Upazila,Kushtia",
    "Khulna Sadar Upazila,Khulna",
    "Kishoreganj Sadar Upazila,Kishoreganj",
    "Kishoreganj Upazila,Nilphamari",
    "Kotalipara Upazila,Gopalganj",
    "Kotchandpur Upazila,Jhenaidah",
    "Koyra Upazila,Khulna",
    "Kulaura Upazila,Moulvibazar",
    "Kuliarchar Upazila,Kishoreganj",
    "Kumarkhali Upazila,Kushtia",
    "Kurigram Sadar Upazila,Kurigram",
    "Kushtia Sadar Upazila,Kushtia",
    "Kutubdia Upazila,Cox_s Bazar",
    "Lakhai Upazila,Habiganj",
    "Laksam Upazila,Cumilla",
    "Lakshmichhari Upazila,Khagrachhari",
    "Lakshmipur Sadar Upazila,Lakshmipur",
    "Lalmai Upazila,Cumilla",
    "Lalmohan Upazila,Bhola",
    "Lalmonirhat Sadar Upazila,Lalmonirhat",
    "Lalpur Upazila,Natore",
    "Lama Upazila,Bandarban",
    "Langadu Upazila,Rangamati",
    "Lohagara Upazila,Chattogram",
    "Lohagara Upazila,Narail",
    "Lohajang Upazila,Munshiganj",
    "Madan Upazila,Netrokona",
    "Madarganj Upazila,Jamalpur",
    "Madaripur Sadar Upazila,Madaripur",
    "Madhabpur Upazila,Habiganj",
    "Madhukhali Upazila,Faridpur",
    "Madhupur Upazila,Tangail",
    "Madhyanagar Upazila,Sunamganj",
    "Magura Sadar Upazila,Magura",
    "Mahalchhari Upazila,Khagrachhari",
    "Maheshkhali Upazila,Cox_s Bazar",
    "Maheshpur Upazila,Jhenaidah",
    "Manda Upazila,Naogaon",
    "Manikchhari Upazila,Khagrachhari",
    "Manikgonj Sadar Upazila,Manikganj",
    "Manirampur Upazila,Jashore",
    "Manpura Upazila,Bhola",
    "Mathbaria Upazila,Pirojpur",
    "Matiranga Upazila,Khagrachhari",
    "Matlab Dakshin Upazila,Chandpur",
    "Matlab Uttar Upazila,Chandpur",
    "Meghna Upazila,Cumilla",
    "Mehendiganj Upazila,Barishal",
    "Meherpur Sadar Upazila,Meherpur",
    "Melandaha Upazila,Jamalpur",
    "Mirpur Upazila,Kushtia",
    "Mirsharai Upazila,Chattogram",
    "Mirzaganj Upazila,Patuakhali",
    "Mirzapur Upazila,Tangail",
    "Mithamain Upazila,Kishoreganj",
    "Mithapukur Upazila,Rangpur",
    "Mohadevpur Upazila,Naogaon",
    "Mohammadpur Upazila,Magura",
    "Mohanganj Upazila,Netrokona",
    "Mohanpur Upazila,Rajshahi",
    "Mollahat Upazila,Bagerhat",
    "Mongla Upazila,Bagerhat",
    "Monohardi Upazila,Narsingdi",
    "Monohargonj Upazila,Cumilla",
    "Morrelganj Upazila,Bagerhat",
    "Moulvibazar Sadar Upazila,Moulvibazar",
    "Mujibnagar Upazila,Meherpur",
    "Muksudpur Upazila,Gopalganj",
    "Muktagachha Upazila,Mymensingh",
    "Muladi Upazila,Barishal",
    "Munshiganj Sadar Upazila,Munshiganj",
    "Muradnagar Upazila,Cumilla",
    "Mymensingh Sadar Upazila,Mymensingh",
    "Nabiganj Upazila,Habiganj",
    "Nabinagar Upazila,Brahmanbaria",
    "Nachole Upazila,Chapai Nawabganj",
    "Nagarkanda Upazila,Faridpur",
    "Nagarpur Upazila,Tangail",
    "Nageshwari Upazila,Kurigram",
    "Naikhongchhari Upazila,Bandarban",
    "Nakla Upazila,Sherpur",
    "Nalchity Upazila,Jhalokati",
    "Naldanga Upazila,Natore",
    "Nalitabari Upazila,Sherpur",
    "Nandail Upazila,Mymensingh",
    "Nandigram Upazila,Bogura",
    "Nangalkot Upazila,Cumilla",
    "Naniyachar Upazila,Rangamati",
    "Naogaon Sadar Upazila,Naogaon",
    "Narail Sadar Upazila,Narail",
    "Narayanganj Sadar Upazila,Narayanganj",
    "Naria Upazila,Shariatpur",
    "Narsingdi Sadar Upazila,Narsingdi",
    "Nasirnagar Upazila,Brahmanbaria",
    "Natore Sadar Upazila,Natore",
    "Nawabganj Sadar Upazila,Chapai Nawabganj",
    "Nawabganj Upazila,Dhaka",
    "Nawabganj Upazila,Dinajpur",
    "Nazirpur Upazila,Pirojpur",
    "Nesarabad (Swarupkati) Upazila,Pirojpur",
    "Netrokona Sadar Upazila,Netrokona",
    "Niamatpur Upazila,Naogaon",
    "Nikli Upazila,Kishoreganj",
    "Nilphamari Sadar Upazila,Nilphamari",
    "Noakhali Sadar Upazila,Noakhali",
    "Osmani Nagar Upazila,Sylhet",
    "Paba Upazila,Rajshahi",
    "Pabna Sadar Upazila,Pabna",
    "Paikgachha Upazila,Khulna",
    "Pakundia Upazila,Kishoreganj",
    "Palash Upazila,Narsingdi",
    "Palashbari Upazila,Gaibandha",
    "Panchagarh Sadar Upazila,Panchagarh",
    "Panchbibi Upazila,Joypurhat",
    "Panchhari Upazila,Khagrachhari",
    "Pangsha Upazila,Rajbari",
    "Parbatipur Upazila,Dinajpur",
    "Parshuram Upazila,Feni",
    "Patgram Upazila,Lalmonirhat",
    "Patharghata Upazila,Barguna",
    "Patiya Upazila,Chattogram",
    "Patnitala Upazila,Naogaon",
    "Patuakhali Sadar Upazila,Patuakhali",
    "Pekua Upazila,Cox_s Bazar",
    "Phulbari Upazila,Dinajpur",
    "Phulbari Upazila,Kurigram",
    "Phulchhari Upazila,Gaibandha",
    "Phulpur Upazila,Mymensingh",
    "Phultala Upazila,Khulna",
    "Pirgachha Upazila,Rangpur",
    "Pirganj Upazila,Rangpur",
    "Pirganj Upazila,Thakurgaon",
    "Pirojpur Sadar Upazila,Pirojpur",
    "Porsha Upazila,Naogaon",
    "Purbadhala Upazila,Netrokona",
    "Puthia Upazila,Rajshahi",
    "Raiganj Upazila,Sirajganj",
    "Raipur Upazila,Lakshmipur",
    "Raipura Upazila,Narsingdi",
    "Rajapur Upazila,Jhalokati",
    "Rajarhat Upazila,Kurigram",
    "Rajasthali Upazila,Rangamati",
    "Rajbari Sadar Upazila,Rajbari",
    "Rajnagar Upazila,Moulvibazar",
    "Rajoir Upazila,Madaripur",
    "Ramganj Upazila,Lakshmipur",
    "Ramgarh Upazila,Khagrachhari",
    "Ramgati Upazila,Lakshmipur",
    "Rampal Upazila,Bagerhat",
    "Ramu Upazila,Cox_s Bazar",
    "Rangabali Upazila,Patuakhali",
    "Rangamati Sadar Upazila,Rangamati",
    "Rangpur Sadar Upazila,Rangpur",
    "Rangunia Upazila,Chattogram",
    "Raninagar Upazila,Naogaon",
    "Ranisankail Upazila,Thakurgaon",
    "Raomari Upazila,Kurigram",
    "Raozan Upazila,Chattogram",
    "Rowangchhari Upazila,Bandarban",
    "Ruma Upazila,Bandarban",
    "Rupganj Upazila,Narayanganj",
    "Rupsha Upazila,Khulna",
    "Sadarpur Upazila,Faridpur",
    "Sadullapur Upazila,Gaibandha",
    "Saidpur Upazila,Nilphamari",
    "Sakhipur Upazila,Tangail",
    "Saltha Upazila,Faridpur",
    "Sandwip Upazila,Chattogram",
    "Santhia Upazila,Pabna",
    "Sapahar Upazila,Naogaon",
    "Sarail Upazila,Brahmanbaria",
    "Sarankhola Upazila,Bagerhat",
    "Sariakandi Upazila,Bogura",
    "Sarishabari Upazila,Jamalpur",
    "Satkania Upazila,Chattogram",
    "Satkhira Sadar Upazila,Satkhira",
    "Saturia Upazila,Manikganj",
    "Savar Upazila,Dhaka",
    "Senbagh Upazila,Noakhali",
    "Shahjadpur Upazila,Sirajganj",
    "Shahrasti Upazila,Chandpur",
    "Shailkupa Upazila,Jhenaidah",
    "Shajahanpur Upazila,Bogura",
    "Shalikha Upazila,Magura",
    "Shantiganj Upazila,Sunamganj",
    "Shariatpur Sadar Upazila,Shariatpur",
    "Sharsha Upazila,Jashore",
    "Shayestaganj Upazila,Habiganj",
    "Sherpur Sadar Upazila,Sherpur",
    "Sherpur Upazila,Bogura",
    "Shibchar Upazila,Madaripur",
    "Shibganj Upazila,Bogura",
    "Shibganj Upazila,Chapai Nawabganj",
    "Shibpur Upazila,Narsingdi",
    "Shivalaya Upazila,Manikganj",
    "Shyamnagar Upazila,Satkhira",
    "Singair Upazila,Manikganj",
    "Singra Upazila,Natore",
    "Sirajdikhan Upazila,Munshiganj",
    "Sirajganj Sadar Upazila,Sirajganj",
    "Sitakunda Upazila,Chattogram",
    "Sonagazi Upazila,Feni",
    "Sonaimuri Upazila,Noakhali",
    "Sonargaon Upazila,Narayanganj",
    "Sonatola Upazila,Bogura",
    "Sreebardi Upazila,Sherpur",
    "Sreemangal Upazila,Moulvibazar",
    "Sreenagar Upazila,Munshiganj",
    "Sreepur Upazila,Gazipur",
    "Sreepur Upazila,Magura",
    "Subarnachar Upazila,Noakhali",
    "Sughatta Upazila,Gaibandha",
    "Sujanagar Upazila,Pabna",
    "Sullah Upazila,Sunamganj",
    "Sunamganj Sadar Upazila,Sunamganj",
    "Sundarganj Upazila,Gaibandha",
    "Sylhet Sadar Upazila,Sylhet",
    "Tahirpur Upazila,Sunamganj",
    "Tala Upazila,Satkhira",
    "Taltali Upazila,Barguna",
    "Tangail Sadar Upazila,Tangail",
    "Tanore Upazila,Rajshahi",
    "Tara Khanda Upazila,Mymensingh",
    "Taraganj Upazila,Rangpur",
    "Tarail Upazila,Kishoreganj",
    "Tarash Upazila,Sirajganj",
    "Tazumuddin Upazila,Bhola",
    "Teknaf Upazila,Cox_s Bazar",
    "Terokhada Upazila,Khulna",
    "Tetulia Upazila,Panchagarh",
    "Thakurgaon Sadar Upazila,Thakurgaon",
    "Thanchi Upazila,Bandarban",
    "Titas Upazila,Cumilla",
    "Tongibari Upazila,Munshiganj",
    "Trishal Upazila,Mymensingh",
    "Tungipara Upazila,Gopalganj",
    "Ukhia Upazila,Cox_s Bazar",
    "Ulipur Upazila,Kurigram",
    "Ullahpara Upazila,Sirajganj",
    "Wazirpur Upazila,Barishal",
    "Zajira Upazila,Shariatpur",
    "Zakiganj Upazila,Sylhet",
]



if len(CLASS_NAMES) != NUM_CLASSES:
    raise ValueError(
        "Class count mismatch. "
        f"NUM_CLASSES={NUM_CLASSES}, "
        f"but CLASS_NAMES contains {len(CLASS_NAMES)} classes."
    )


if len(set(CLASS_NAMES)) != NUM_CLASSES:
    raise ValueError(
        "Duplicate class names were found inside CLASS_NAMES."
    )


if abs(sum(VOTING_WEIGHTS) - 1.0) > 1e-8:
    raise ValueError(
        "Soft-voting weights must add up to 1.0."
    )



def find_repository_file(
    repository_files: list[str],
    expected_filename: str,
) -> str:
    """
    Find the expected model file in the Hugging Face repository.

    It supports:
    - Files stored in the repository root
    - Files stored inside a subfolder
    """

    if expected_filename in repository_files:
        return expected_filename

    matches = [
        file_path
        for file_path in repository_files
        if Path(file_path).name.lower()
        == expected_filename.lower()
    ]

    if len(matches) == 1:
        return matches[0]

    if not matches:
        available_models = [
            file_path
            for file_path in repository_files
            if Path(file_path).suffix.lower()
            in {
                ".pth",
                ".pt",
                ".bin",
            }
        ]

        available_text = (
            "\n".join(available_models)
            if available_models
            else "No PyTorch model files found."
        )

        raise FileNotFoundError(
            "Required model file was not found "
            "in the Hugging Face repository.\n\n"
            f"Expected file:\n{expected_filename}\n\n"
            f"Available model files:\n{available_text}"
        )

    raise RuntimeError(
        "More than one repository file matched:\n"
        f"{expected_filename}\n\n"
        + "\n".join(matches)
    )


def download_model_checkpoints() -> tuple[Path, Path]:
    """
    Locate and download the selected ViT and ConvNeXt
    checkpoint files from Hugging Face.
    """

    api = HfApi()

    repository_files = api.list_repo_files(
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        revision=MODEL_REVISION,
    )

    vit_repository_path = find_repository_file(
        repository_files=repository_files,
        expected_filename=VIT_MODEL_FILENAME,
    )

    convnext_repository_path = find_repository_file(
        repository_files=repository_files,
        expected_filename=CONVNEXT_MODEL_FILENAME,
    )

    print(
        "✅ ViT repository file:",
        vit_repository_path,
    )

    print(
        "✅ ConvNeXt repository file:",
        convnext_repository_path,
    )

    vit_checkpoint = Path(
        hf_hub_download(
            repo_id=MODEL_REPO_ID,
            repo_type="model",
            revision=MODEL_REVISION,
            filename=vit_repository_path,
        )
    )

    convnext_checkpoint = Path(
        hf_hub_download(
            repo_id=MODEL_REPO_ID,
            repo_type="model",
            revision=MODEL_REVISION,
            filename=convnext_repository_path,
        )
    )

    return (
        vit_checkpoint,
        convnext_checkpoint,
    )



def build_vit_base_custom_head(
    num_classes: int,
    dropout: float = 0.30,
    hidden_ratio: float = 0.50,
) -> ViTForImageClassification:
    """
    Build the ViT-Base model architecture used during training.

    Backbone:
        ViT-Base Patch16 224

    Classification head:
        LayerNorm
        Dropout
        Linear
        GELU
        Dropout
        Linear
    """

    config = ViTConfig(
        image_size=224,
        patch_size=16,
        num_channels=3,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        qkv_bias=True,
        encoder_stride=16,
        num_labels=num_classes,
    )

    model = ViTForImageClassification(
        config
    )

    hidden_size = (
        model.config.hidden_size
    )

    middle_size = int(
        hidden_size
        * hidden_ratio
    )

    model.classifier = nn.Sequential(
        nn.LayerNorm(
            hidden_size
        ),
        nn.Dropout(
            dropout
        ),
        nn.Linear(
            hidden_size,
            middle_size
        ),
        nn.GELU(),
        nn.Dropout(
            dropout
        ),
        nn.Linear(
            middle_size,
            num_classes
        ),
    )

    return model



class ConvNeXtCustom(nn.Module):
    """
    ConvNeXt backbone with the custom MLP classification head
    used during model training.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        dropout: float = 0.30,
        hidden_ratio: float = 0.50,
    ) -> None:
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            global_pool="",
        )

        self.hidden = getattr(
            self.backbone,
            "num_features",
            None,
        )

        if self.hidden is None:
            raise ValueError(
                "Could not find "
                "ConvNeXt backbone.num_features."
            )

        middle_size = int(
            self.hidden
            * hidden_ratio
        )

        self.pool = (
            nn.AdaptiveAvgPool2d(1)
        )

        self.head = nn.Sequential(
            nn.LayerNorm(
                self.hidden
            ),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                self.hidden,
                middle_size
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                middle_size,
                num_classes
            ),
        )

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Run the ConvNeXt forward pass."""

        features = (
            self.backbone
            .forward_features(
                images
            )
        )

        if features.dim() == 4:

            if (
                features.shape[1]
                != self.hidden
                and features.shape[-1]
                == self.hidden
            ):
                features = (
                    features
                    .permute(
                        0,
                        3,
                        1,
                        2,
                    )
                    .contiguous()
                )

            features = (
                self.pool(
                    features
                )
                .flatten(1)
            )

        elif features.dim() == 2:
            pass

        else:
            raise RuntimeError(
                "Unexpected ConvNeXt feature shape: "
                f"{tuple(features.shape)}"
            )

        return self.head(
            features
        )


def build_convnext_custom_head(
    num_classes: int,
    variant: str = "tiny",
    dropout: float = 0.30,
    hidden_ratio: float = 0.50,
) -> ConvNeXtCustom:
    """
    Build the selected ConvNeXt model.

    Current version:
        ConvNeXt-Tiny
    """

    model_names = {
        "tiny": "convnext_tiny",
        "base": "convnext_base",
    }

    if variant not in model_names:
        raise ValueError(
            "Unsupported ConvNeXt variant: "
            f"{variant}"
        )

    return ConvNeXtCustom(
        model_name=model_names[variant],
        num_classes=num_classes,
        dropout=dropout,
        hidden_ratio=hidden_ratio,
    )


def read_state_dictionary(
    checkpoint_path: Path,
) -> dict[str, torch.Tensor]:
    """
    Read a PyTorch state dictionary.

    Supported checkpoint formats:
    - Plain model.state_dict()
    - {"state_dict": ...}
    - {"model_state_dict": ...}
    - {"model": ...}
    """

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise TypeError(
            "Invalid checkpoint format:\n"
            f"{checkpoint_path}"
        )

    for key in (
        "state_dict",
        "model_state_dict",
        "model",
    ):
        nested_dictionary = (
            checkpoint.get(key)
        )

        if isinstance(
            nested_dictionary,
            dict,
        ):
            checkpoint = nested_dictionary
            break

    cleaned_state_dictionary = {}

    for key, value in checkpoint.items():

        cleaned_key = str(key)

        if cleaned_key.startswith(
            "module."
        ):
            cleaned_key = cleaned_key[
                len("module.") :
            ]

        cleaned_state_dictionary[
            cleaned_key
        ] = value

    return cleaned_state_dictionary



def convert_legacy_vit_state_dictionary(
    state_dictionary: dict[str, torch.Tensor],
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    """
    Convert older Hugging Face ViT checkpoint keys into
    the naming format expected by newer Transformers versions.

    Old examples:
        vit.encoder.layer.0.attention.attention.query.weight
        vit.encoder.layer.0.attention.output.dense.weight
        vit.encoder.layer.0.intermediate.dense.weight
        vit.encoder.layer.0.output.dense.weight

    New examples:
        vit.layers.0.attention.q_proj.weight
        vit.layers.0.attention.o_proj.weight
        vit.layers.0.mlp.fc1.weight
        vit.layers.0.mlp.fc2.weight
    """

    checkpoint_uses_legacy_keys = any(
        key.startswith("vit.encoder.layer.")
        for key in state_dictionary
    )

    model_uses_new_keys = any(
        key.startswith("vit.layers.")
        for key in model.state_dict()
    )

    if not (
        checkpoint_uses_legacy_keys
        and model_uses_new_keys
    ):
        return state_dictionary

    replacements = (
        (
            "vit.encoder.layer.",
            "vit.layers.",
        ),
        (
            ".attention.attention.query.",
            ".attention.q_proj.",
        ),
        (
            ".attention.attention.key.",
            ".attention.k_proj.",
        ),
        (
            ".attention.attention.value.",
            ".attention.v_proj.",
        ),
        (
            ".attention.output.dense.",
            ".attention.o_proj.",
        ),
        (
            ".intermediate.dense.",
            ".mlp.fc1.",
        ),
        (
            ".output.dense.",
            ".mlp.fc2.",
        ),
    )

    converted_dictionary: dict[
        str,
        torch.Tensor,
    ] = {}

    converted_count = 0

    for original_key, tensor in state_dictionary.items():

        converted_key = original_key

        for old_text, new_text in replacements:
            converted_key = converted_key.replace(
                old_text,
                new_text,
            )

        if converted_key != original_key:
            converted_count += 1

        if converted_key in converted_dictionary:
            raise RuntimeError(
                "Duplicate key created during ViT "
                "checkpoint conversion:\n"
                f"{converted_key}"
            )

        converted_dictionary[
            converted_key
        ] = tensor

    print(
        "✅ Converted legacy ViT checkpoint keys:",
        converted_count,
    )

    return converted_dictionary


def load_checkpoint_strictly(
    model: nn.Module,
    checkpoint_path: Path,
    model_name: str,
) -> nn.Module:
    """
    Load model weights with strict validation.

    Legacy ViT checkpoint names are converted automatically
    when the current Transformers model expects new names.
    """

    state_dictionary = read_state_dictionary(
        checkpoint_path
    )

    if model_name.lower().startswith("vit"):

        state_dictionary = (
            convert_legacy_vit_state_dictionary(
                state_dictionary=state_dictionary,
                model=model,
            )
        )

    try:
        model.load_state_dict(
            state_dictionary,
            strict=True,
        )

    except RuntimeError as error:
        raise RuntimeError(
            f"{model_name} checkpoint does not "
            "match the pipeline architecture.\n\n"
            f"Checkpoint:\n{checkpoint_path}\n\n"
            f"Original error:\n{error}"
        ) from error

    print(
        f"✅ {model_name} checkpoint loaded strictly."
    )

    return model

def crop_text_area_pil(
    image: Image.Image,
    padding: int = 6,
) -> Image.Image:
    """
    Detect and crop the handwritten text area.

    This function is used only when:
        USE_CROP = True
    """

    rgb_array = np.array(
        image.convert("RGB")
    )

    gray_array = np.array(
        image.convert("L")
    )

    threshold = np.percentile(
        gray_array,
        85,
    )

    mask = (
        gray_array
        < threshold
    )

    coordinates = np.argwhere(
        mask
    )

    if coordinates.size == 0:
        return image.convert(
            "RGB"
        )

    y_min, x_min = (
        coordinates.min(
            axis=0
        )
    )

    y_max, x_max = (
        coordinates.max(
            axis=0
        )
    )

    y_min = max(
        0,
        int(y_min) - padding,
    )

    x_min = max(
        0,
        int(x_min) - padding,
    )

    y_max = min(
        rgb_array.shape[0] - 1,
        int(y_max) + padding,
    )

    x_max = min(
        rgb_array.shape[1] - 1,
        int(x_max) + padding,
    )

    if (
        (y_max - y_min) < 20
        or (x_max - x_min) < 20
    ):
        return image.convert(
            "RGB"
        )

    cropped_image = rgb_array[
        y_min : y_max + 1,
        x_min : x_max + 1,
    ]

    return Image.fromarray(
        cropped_image
    )


def prepare_inference_image(
    image: Image.Image,
) -> Image.Image:
    """
    Convert the uploaded image to RGB.

    Cropping remains disabled because:
        USE_CROP = False
    """

    rgb_image = image.convert(
        "RGB"
    )

    if USE_CROP:
        return crop_text_area_pil(
            rgb_image,
            padding=6,
        )

    return rgb_image


INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.Lambda(
            prepare_inference_image
        ),

        transforms.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            )
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=NORM_MEAN,
            std=NORM_STD,
        ),
    ]
)


def clean_display_name(
    name: str,
) -> str:
    """
    Convert dataset-safe names into normal display names.

    Example:
        Cox_s Bazar
        becomes
        Cox's Bazar
    """

    return (
        name
        .replace(
            "_s",
            "'s"
        )
        .replace(
            "_",
            " "
        )
        .strip()
    )


def split_class_label(
    class_label: str,
) -> tuple[str, str]:
    """
    Split one class label into Upazila and District.

    Example input:
        Cumilla Sadar Upazila,Cumilla

    Returned output:
        Cumilla Sadar
        Cumilla
    """

    if "," not in class_label:
        return (
            clean_display_name(
                class_label
            ),
            "Unknown",
        )

    upazila, district = (
        class_label.rsplit(
            ",",
            1,
        )
    )

    upazila = clean_display_name(
        upazila
    )

    district = clean_display_name(
        district
    )

    suffix = " Upazila"

    if upazila.endswith(
        suffix
    ):
        upazila = upazila[
            : -len(suffix)
        ].strip()

    return (
        upazila,
        district,
    )


class UpazilaDistrictPredictor:
    """
    Download, load and run the ViT–ConvNeXt ensemble.
    """

    def __init__(self) -> None:

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        (
            vit_checkpoint,
            convnext_checkpoint,
        ) = download_model_checkpoints()

        print(
            "✅ ViT local checkpoint:",
            vit_checkpoint,
        )

        print(
            "✅ ConvNeXt local checkpoint:",
            convnext_checkpoint,
        )

        self.vit_model = (
            build_vit_base_custom_head(
                num_classes=NUM_CLASSES,
                dropout=0.30,
                hidden_ratio=0.50,
            )
        )

        self.convnext_model = (
            build_convnext_custom_head(
                num_classes=NUM_CLASSES,
                variant=CONVNEXT_VARIANT,
                dropout=0.30,
                hidden_ratio=0.50,
            )
        )

        self.vit_model = (
            load_checkpoint_strictly(
                model=self.vit_model,
                checkpoint_path=vit_checkpoint,
                model_name="ViT-Base",
            )
        )

        self.convnext_model = (
            load_checkpoint_strictly(
                model=self.convnext_model,
                checkpoint_path=convnext_checkpoint,
                model_name="ConvNeXt-Tiny",
            )
        )

        self.vit_model = (
            self.vit_model
            .to(self.device)
            .eval()
        )

        self.convnext_model = (
            self.convnext_model
            .to(self.device)
            .eval()
        )

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Predict the Upazila–District pair from one image.
        """

        if not isinstance(
            image,
            Image.Image,
        ):
            raise TypeError(
                "The input image must "
                "be a PIL Image."
            )

        image_tensor = (
            INFERENCE_TRANSFORM(
                image
            )
            .unsqueeze(0)
            .to(self.device)
        )

        vit_output = self.vit_model(
            image_tensor
        )

        vit_logits = (
            vit_output.logits
            if hasattr(
                vit_output,
                "logits",
            )
            else vit_output
        )

        convnext_logits = (
            self.convnext_model(
                image_tensor
            )
        )

        combined_logits = (
            VIT_WEIGHT
            * vit_logits
            +
            CONVNEXT_WEIGHT
            * convnext_logits
        )

        probabilities = torch.softmax(
            combined_logits,
            dim=1,
        )[0]

        safe_top_k = max(
            1,
            min(
                int(top_k),
                NUM_CLASSES,
            ),
        )

        (
            top_probabilities,
            top_indices,
        ) = torch.topk(
            probabilities,
            k=safe_top_k,
        )

        predicted_index = int(
            top_indices[0].item()
        )

        predicted_label = (
            CLASS_NAMES[
                predicted_index
            ]
        )

        (
            upazila,
            district,
        ) = split_class_label(
            predicted_label
        )

        top_predictions = []

        for (
            probability,
            index_tensor,
        ) in zip(
            top_probabilities,
            top_indices,
        ):

            class_index = int(
                index_tensor.item()
            )

            class_label = (
                CLASS_NAMES[
                    class_index
                ]
            )

            (
                candidate_upazila,
                candidate_district,
            ) = split_class_label(
                class_label
            )

            top_predictions.append(
                {
                    "class_index": (
                        class_index
                    ),

                    "class_label": (
                        class_label
                    ),

                    "upazila": (
                        candidate_upazila
                    ),

                    "district": (
                        candidate_district
                    ),

                    "confidence": round(
                        float(
                            probability.item()
                        )
                        * 100.0,
                        4,
                    ),
                }
            )

        return {
            "upazila": upazila,

            "district": district,

            "confidence": round(
                float(
                    top_probabilities[
                        0
                    ].item()
                )
                * 100.0,
                4,
            ),

            "model": (
                f"Weighted Soft Voting {MODEL_VERSION} "
                "(ViT-Base + ConvNeXt-Tiny)"
            ),

            "class_index": (
                predicted_index
            ),

            "class_label": (
                predicted_label
            ),

            "top_predictions": (
                top_predictions
            ),
        }



@lru_cache(maxsize=1)
def get_predictor(
) -> UpazilaDistrictPredictor:
    """
    Load the two models only once.

    Future predictions reuse the loaded models.
    """

    return UpazilaDistrictPredictor()


def predict_upazila_district(
    image: Image.Image,
) -> dict[str, Any]:
    """
    Public function that app.py will use later.
    """

    return (
        get_predictor()
        .predict(
            image=image,
            top_k=5,
        )
    )


__all__ = ["predict_upazila_district"]



def run_pipeline_startup_test() -> None:
    """
    Download and strictly load both model checkpoints.

    This test does not require an input image.
    """

    print(
        "=" * 68
    )

    print(
        "STEP 1 — INFERENCE PIPELINE STARTUP TEST"
    )

    print(
        "=" * 68
    )

    print(
        "Repository:",
        MODEL_REPO_ID,
    )

    print(
        "Revision:",
        MODEL_REVISION,
    )

    print(
        "Model version:",
        MODEL_VERSION,
    )

    print(
        "Class count:",
        len(CLASS_NAMES),
    )

    print(
        "Device:",
        (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    print(
        "ViT checkpoint:",
        VIT_MODEL_FILENAME,
    )

    print(
        "ConvNeXt checkpoint:",
        CONVNEXT_MODEL_FILENAME,
    )

    print(
        "Downloading and loading models..."
    )

    predictor = get_predictor()

    print()

    print(
        "✅ ViT checkpoint loaded successfully."
    )

    print(
        "✅ ConvNeXt checkpoint loaded successfully."
    )

    print(
        "✅ Models are ready on:",
        predictor.device,
    )

    print(
        "✅ predict_upazila_district(image) "
        "is ready."
    )

    print(
        "=" * 68
    )


if __name__ == "__main__":
    run_pipeline_startup_test()
