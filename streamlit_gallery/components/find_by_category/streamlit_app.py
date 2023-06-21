import streamlit as st

from streamlit_ace import st_ace, KEYBINDINGS, LANGUAGES, THEMES
from streamlit_gallery.utils.readme import readme

import greenplumpython as gp
import os

from PIL import Image
import requests
from io import BytesIO

import numpy as np

db = gp.database(uri="postgres://gpadmin:changeme@35.225.47.84:5432/warehouse")

gp.config.print_sql = True

# Get repr of Grenplum operator vector cosine distance
cosine_distance = gp.operator("<=>") 

vector = gp.type_("vector", modifier=512)

fashion_images = db.create_dataframe(table_name="product_embeddings", schema="fashion")
images_styles = db.create_dataframe(table_name="image_styles", schema="fashion")

GENDER = ("NoSpecified", "Women", "Men", "Girls", "Boys", "Unisex")
MASTERCATEGORY = ("NoSpecified", "Free Items", "Apparel", "Sporting Goods", "Footwear", "Personal Care", "Accessories", "Home")
SUBCATEGORY = {
    "NoSpecified": ("NoSpecified", ""),
    "Accessories": (
        "Accessories", "Bags", "Belts", "Cufflinks", "Eyewear", "Gloves", "Heaswear", "Jewellery", "Mufflers", "Perfumes", "Scarves",
        "Shoe Accessories", "Socks", "Sports Accessories", "Stoles", "Ties", "Umbrellas", "Wallets", "Watches", "Water Bottle",
    ), 
    "Apparel": (
        "Apparel Set", "Bottomwear", "Dress", "Innerwear", "Loungewear and Nightwear", "Saree", "Socks", "Topwear", 
    ), 
    "Footwear": ("Flip Flops", "Sandal", "Shoes"), 
    "Free Items": ("Free Gifts", "Vouchers"), 
    "Home": ("Home Furnishing", ""),
    "Personal Care": (
        "Bath and Body", "Beauty Accessories", "Eyes", "Fragrance", "Hair", "Lips", "Makeup", "Nails", "Perfumes", "Skin", "Skin Care",
    ),
    "Sporting Goods": ("Sports Equipment", "Wristbands"),
}
ARTICLETTYPE = {
    "NoSpecified": ("NoSpecified", ""),
    "Accessories": (
        "Accessory Gift Set", "Hair Accessory", "Key chain", "Messenger Bag", "Travel Accessory", "Water Bottle", "Clothing Set", "Kurta Sets", "Swimwear",
    ),
    "Bags": (
        'Backpacks',
        'Clutches',
        'Duffel Bag',
        'Handbags',
        'Laptop Bag',
        'Messenger Bag',
        'Mobile Pouch',
        'Rucksacks',
        'Tablet Sleeve',
        'Travel Accessory',
        'Trolley Bag',
        'Waist Pouch',
        'Wallets'
    ),
    "Bath and Body": (
        'Body Lotion', 'Body Wash and Scrub', 'Nail Essentials',
    ),
    "Beauty Accessories": ("Beauty Accessories"),
    "Belts": ("Belts", "Tshirts"),
    "Bottomwear": (
        'Capris',
        'Churidar',
        'Jeans',
        'Jeggings',
        'Leggings',
        'Patiala',
        'Rain Trousers',
        'Salwar',
        'Salwar and Dupatta',
        'Shorts',
        'Skirts',
        'Stockings',
        'Swimwear',
        'Tights',
        'Track Pants',
        'Tracksuits',
        'Trousers',
    ),
    "Cufflinks": ("Cufflinks", "Ties and Cufflinks"),
    "Dress": ("Dresses", "Jumpsuit"),
    "Eyes": ('Eyeshadow', 'Kajal and Eyeliner', 'Mascara'),
    "Eyewear": ("Sunglasses", ""),
    "Flip Flops": ("Flip Flops", ""),
    "Fragrance": ('Deodorant', 'Fragrance Gift Set', 'Perfume and Body Mist'),
    "Free Gifts": (
        'Backpacks',
        'Clutches',
        'Free Gifts',
        'Handbags',
        'Laptop Bag',
        'Scarves',
        'Ties',
        'Wallets'
    ),
    "Gloves": ("Gloves", ""),
    "Hair": ("Hair Colour", ""),
    "Headwear": ('Caps', 'Hat', 'Headband'),
    "Home Furnishing": ("Cushion Covers", ""),
    "Innerwear": (
        'Boxers',
        'Bra',
        'Briefs',
        'Camisoles',
        'Innerwear Vests',
        'Shapewear',
        'Trunk',
    ),
    "Jewellery": (
        'Bangle',
        'Bracelet',
        'Earrings',
        'Jewellery Set',
        'Necklace and Chains',
        'Pendant',
        'Ring',
    ),
    "Lips": ('Lip Care', 'Lip Gloss', 'Lip Liner', 'Lip Plumper', 'Lipstick'),
    "Loungewear and Nightwear": (
        'Baby Dolls',
        'Bath Robe',
        'Lounge Pants',
        'Lounge Shorts',
        'Lounge Tshirts',
        'Nightdress',
        'Night suits',
        'Robe',
        'Shorts',
    ),
    "Makeup": (
        'Compact',
        'Concealer',
        'Eyeshadow',
        'Foundation and Primer',
        'Highlighter and Blush',
        'Kajal and Eyeliner',
        'Makeup Remover',
    ),
    "Mufflers": ("Mufflers", ""),
    "Nails": ("Nail Polish", ""),
    "Perfumes": ("Perfume and Body Mist", ""), 
    "Sandal": ("Flip Flops", "Sandals", "Sports Sandals"),
    "Saree": ("Sarees", ""),
    "Scarves": ("Scarves", ""),
    "Shoe Accessories": (
        "Shoe Accessories", "Shoe Laces"
    ),
    "Shoes": (
        'Casual Shoes', 'Flats', 'Formal Shoes', 'Heels', 'Sandals', 'Sports Shoes'
    ),
    "Skin": (
        'Body Lotion', 'Face Moisturisers', 'Face Serum and Gel', 'Mask and Peel'
    ),
    "Skin Care": (
        'Eye Cream',
        'Face Moisturisers',
        'Face Scrub and Exfoliator',
        'Face Wash and Cleanser',
        'Mask and Peel',
        'Mens Grooming Kit',
        'Sunscreen',
        'Toner',
    ),
    "Socks": ("Booties", "Socks"),
    "Sports Accessories": ("Wristbands"),
    "Sports Equipment": ("Basketballs", "Footballs"),
    "Stoles": ("Stoles", ""),
    "Ties": ("Ties", ""),
    "Topwear": (
        'Belts',
        'Blazers',
        'Dresses',
        'Dupatta',
        'Jackets',
        'Kurtas',
        'Kurtis',
        'Lehenga Choli',
        'Nehru Jackets',
        'Rain Jacket',
        'Rompers',
        'Shirts',
        'Shrug',
        'Suits',
        'Suspenders',
        'Sweaters',
        'Sweatshirts',
        'Tops',
        'Tshirts',
        'Tunics',
        'Waistcoat',
    ),
    "Umbrellas": ("Umbrellas", ""),
    "Vouchers":("Ipad", ""),
    "Wallets": ("Wallets", ""), 
    "Watches": ("Watches", ""), 
    "Water Bottle": ("Water Bottle, "),  
    "Wristbands": ("Wristbands", ""),
    "NoSpecified": ("NoSpecified", ""), 
}
BASECOLOUR = (
    'NoSpecified',
    'Grey Melange',
    'Navy Blue',
    'Olive',
    'Lime Green',
    'Mustard',
    'Gold',
    'Multi',
    'Taupe',
    'Cream',
    'Khaki',
    'Magenta',
    'Blue',
    'Nude',
    'Orange',
    'Tan',
    'White',
    'Bronze',
    'Copper',
    'Silver',
    'Maroon',
    'Mauve',
    'Fluorescent Green',
    'Purple',
    'Red',
    'Steel',
    'Peach',
    'Metallic',
    'Brown',
    'Grey',
    'Off White',
    'Rust',
    'Teal',
    'Black',
    'Coffee Brown',
    'Green',
    'Pink',
    'Skin',
    'Beige',
    'Charcoal',
    'Lavender',
    'Mushroom Brown',
    'Rose',
    'Turquoise Blue',
    'Sea Green',
    'Burgundy',
    'Yellow',
 )

SEASON = ("NoSpecified", "Spring", "Summer", "Fall", "Winter")
USAGE = ("NoSpecified", 'Formal', 'Sports', 'Travel', 'Home', 'Casual', 'Ethnic', 'Party', 'Smart Casual')

def main():
    c1, c2 = st.columns([1, 3])

    c1.subheader("Categories")
    gender = c1.selectbox('Gender:', options=GENDER, key="gender")
    masterCategory = c1.selectbox('Product Main Category:', options=MASTERCATEGORY, key="mastercat")
    subCategory = c1.selectbox('Product Sub-Category:', options=SUBCATEGORY[masterCategory], key="subcat")
    articleType = c1.selectbox('Product Type:', options=ARTICLETTYPE[subCategory], key="type")
    baseColour = c1.selectbox('Product Colour:', options=BASECOLOUR, key="colour")
    season = c1.selectbox('Product Season:', options=SEASON, key="season")
    year=c1.text_input("Product Year", value="From 2007 to 2019", key="year")
    usage = c1.selectbox('Product Usage:', options=USAGE, key="usage")
    filter_button = c1.button("Filter")

    if filter_button:
        c2.subheader("Results")
        result = images_styles.where(
        lambda t: (
            (t["gender"] == gender)
            & (t["mastercategory"] == masterCategory) 
            & (t["subcategory"] == subCategory) 
            & (t["articletype"] == articleType) 
            & (t["basecolour"] == baseColour) 
            & (t["season"] == season) 
            & (t["year"] == year) 
            & (t["usage"] == usage) 
        )
        )[:20]    
        images = []
        captions = []
        for row in result:
            response = requests.get(row["link"])
            img = Image.open(BytesIO(response.content))
            images.append(img)
            captions.append(row["productdisplayname"])
        c2.image(images, width=200, caption=captions)


if __name__ == "__main__":
    main()
