import streamlit as st
from multiprocessing import Pool
import requests
from PIL import Image
from io import BytesIO
import os

import greenplumpython as gp

db = gp.database(
    params={
        "host": st.secrets["db_hostname"],
        "dbname": st.secrets["db_name"],
        "user": st.secrets["db_username"],
        "port": st.secrets["db_port"],
        "password": st.secrets["db_password"],
    }
)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

gp.config.print_sql = True

# Get repr of Grenplum operator vector cosine distance
cosine_distance = gp.operator("<=>")

vector = gp.type_("vector", modifier=512)

images_styles = db.create_dataframe(table_name="image_styles", schema="fashion")


# Define Sliders Contents
GENDER = ("Not Specified", "Women", "Men", "Girls", "Boys", "Unisex")
MASTERCATEGORY = (
    "Not Specified",
    "Free Items",
    "Apparel",
    "Sporting Goods",
    "Footwear",
    "Personal Care",
    "Accessories",
    "Home",
)
SUBCATEGORY = {
    "Not Specified": ("Not Specified", ""),
    "Accessories": (
        "Accessories",
        "Bags",
        "Belts",
        "Cufflinks",
        "Eyewear",
        "Gloves",
        "Heaswear",
        "Jewellery",
        "Mufflers",
        "Perfumes",
        "Scarves",
        "Shoe Accessories",
        "Socks",
        "Sports Accessories",
        "Stoles",
        "Ties",
        "Umbrellas",
        "Wallets",
        "Watches",
        "Water Bottle",
        "Not Specified",
    ),
    "Apparel": (
        "Apparel Set",
        "Bottomwear",
        "Dress",
        "Innerwear",
        "Loungewear and Nightwear",
        "Saree",
        "Socks",
        "Topwear",
        "Not Specified",
    ),
    "Footwear": ("Flip Flops", "Sandal", "Shoes", "Not Specified"),
    "Free Items": ("Free Gifts", "Vouchers", "Not Specified"),
    "Home": ("Home Furnishing", "Not Specified"),
    "Personal Care": (
        "Bath and Body",
        "Beauty Accessories",
        "Eyes",
        "Fragrance",
        "Hair",
        "Lips",
        "Makeup",
        "Nails",
        "Perfumes",
        "Skin",
        "Skin Care",
        "Not Specified",
    ),
    "Sporting Goods": ("Sports Equipment", "Wristbands", "Not Specified"),
}
ARTICLETTYPE = {
    "Not Specified": ("Not Specified", "Not Specified"),
    "Accessories": (
        "Not Specified",
        "Accessory Gift Set",
        "Hair Accessory",
        "Key chain",
        "Messenger Bag",
        "Travel Accessory",
        "Water Bottle",
        "Clothing Set",
        "Kurta Sets",
        "Swimwear",
    ),
    "Apparel Set": ("Clothing Set", "Kurta Sets", "Swimwear", "Not Specified"),
    "Bags": (
        "Backpacks",
        "Clutches",
        "Duffel Bag",
        "Handbags",
        "Laptop Bag",
        "Messenger Bag",
        "Mobile Pouch",
        "Rucksacks",
        "Tablet Sleeve",
        "Travel Accessory",
        "Trolley Bag",
        "Waist Pouch",
        "Wallets",
        "Not Specified",
    ),
    "Bath and Body": (
        "Body Lotion",
        "Body Wash and Scrub",
        "Nail Essentials",
        "Not Specified",
    ),
    "Beauty Accessories": ("Beauty Accessories"),
    "Belts": ("Belts", "Tshirts"),
    "Bottomwear": (
        "Capris",
        "Churidar",
        "Jeans",
        "Jeggings",
        "Leggings",
        "Patiala",
        "Rain Trousers",
        "Salwar",
        "Salwar and Dupatta",
        "Shorts",
        "Skirts",
        "Stockings",
        "Swimwear",
        "Tights",
        "Track Pants",
        "Tracksuits",
        "Trousers",
        "Not Specified",
    ),
    "Cufflinks": ("Cufflinks", "Ties and Cufflinks", "Not Specified"),
    "Dress": ("Dresses", "Jumpsuit", "Not Specified"),
    "Eyes": ("Eyeshadow", "Kajal and Eyeliner", "Mascara", "Not Specified"),
    "Eyewear": ("Sunglasses", "Not Specified"),
    "Flip Flops": ("Flip Flops", "Not Specified"),
    "Fragrance": (
        "Deodorant",
        "Fragrance Gift Set",
        "Perfume and Body Mist",
        "Not Specified",
    ),
    "Free Gifts": (
        "Backpacks",
        "Clutches",
        "Free Gifts",
        "Handbags",
        "Laptop Bag",
        "Scarves",
        "Ties",
        "Wallets",
        "Not Specified",
    ),
    "Gloves": ("Gloves", "Not Specified"),
    "Hair": ("Hair Colour", "Not Specified"),
    "Headwear": ("Caps", "Hat", "Headband", "Not Specified"),
    "Home Furnishing": ("Cushion Covers", "Not Specified"),
    "Innerwear": (
        "Boxers",
        "Bra",
        "Briefs",
        "Camisoles",
        "Innerwear Vests",
        "Shapewear",
        "Trunk",
        "Not Specified",
    ),
    "Jewellery": (
        "Bangle",
        "Bracelet",
        "Earrings",
        "Jewellery Set",
        "Necklace and Chains",
        "Pendant",
        "Ring",
        "Not Specified",
    ),
    "Lips": (
        "Lip Care",
        "Lip Gloss",
        "Lip Liner",
        "Lip Plumper",
        "Lipstick",
        "Not Specified",
    ),
    "Loungewear and Nightwear": (
        "Baby Dolls",
        "Bath Robe",
        "Lounge Pants",
        "Lounge Shorts",
        "Lounge Tshirts",
        "Nightdress",
        "Night suits",
        "Robe",
        "Shorts",
        "Not Specified",
    ),
    "Makeup": (
        "Compact",
        "Concealer",
        "Eyeshadow",
        "Foundation and Primer",
        "Highlighter and Blush",
        "Kajal and Eyeliner",
        "Makeup Remover",
        "Not Specified",
    ),
    "Mufflers": ("Mufflers", "Not Specified"),
    "Nails": ("Nail Polish", "Not Specified"),
    "Perfumes": ("Perfume and Body Mist", "Not Specified"),
    "Sandal": ("Flip Flops", "Sandals", "Sports Sandals", "Not Specified"),
    "Saree": ("Sarees", "Not Specified"),
    "Scarves": ("Scarves", "Not Specified"),
    "Shoe Accessories": (
        "Shoe Accessories",
        "Shoe Laces",
        "Not Specified",
    ),
    "Shoes": (
        "Casual Shoes",
        "Flats",
        "Formal Shoes",
        "Heels",
        "Sandals",
        "Sports Shoes",
        "Not Specified",
    ),
    "Skin": (
        "Body Lotion",
        "Face Moisturisers",
        "Face Serum and Gel",
        "Mask and Peel",
        "Not Specified",
    ),
    "Skin Care": (
        "Eye Cream",
        "Face Moisturisers",
        "Face Scrub and Exfoliator",
        "Face Wash and Cleanser",
        "Mask and Peel",
        "Mens Grooming Kit",
        "Sunscreen",
        "Toner",
        "Not Specified",
    ),
    "Socks": ("Booties", "Socks", "Not Specified"),
    "Sports Accessories": ("Wristbands", "Not Specified"),
    "Sports Equipment": ("Basketballs", "Footballs", "Not Specified"),
    "Stoles": ("Stoles", "Not Specified"),
    "Ties": ("Ties", "Not Specified"),
    "Topwear": (
        "Belts",
        "Blazers",
        "Dresses",
        "Dupatta",
        "Jackets",
        "Kurtas",
        "Kurtis",
        "Lehenga Choli",
        "Nehru Jackets",
        "Rain Jacket",
        "Rompers",
        "Shirts",
        "Shrug",
        "Suits",
        "Suspenders",
        "Sweaters",
        "Sweatshirts",
        "Tops",
        "Tshirts",
        "Tunics",
        "Waistcoat",
        "Not Specified",
    ),
    "Umbrellas": ("Umbrellas", "Not Specified"),
    "Vouchers": ("Ipad", "Not Specified"),
    "Wallets": ("Wallets", "Not Specified"),
    "Watches": ("Watches", "Not Specified"),
    "Water Bottle": ("Water Bottle", "Not Specified"),
    "Wristbands": ("Wristbands", "Not Specified"),
    "Not Specified": ("Not Specified", "Not Specified"),
}
BASECOLOUR = (
    "Not Specified",
    "Grey Melange",
    "Navy Blue",
    "Olive",
    "Lime Green",
    "Mustard",
    "Gold",
    "Multi",
    "Taupe",
    "Cream",
    "Khaki",
    "Magenta",
    "Blue",
    "Nude",
    "Orange",
    "Tan",
    "White",
    "Bronze",
    "Copper",
    "Silver",
    "Maroon",
    "Mauve",
    "Fluorescent Green",
    "Purple",
    "Red",
    "Steel",
    "Peach",
    "Metallic",
    "Brown",
    "Grey",
    "Off White",
    "Rust",
    "Teal",
    "Black",
    "Coffee Brown",
    "Green",
    "Pink",
    "Skin",
    "Beige",
    "Charcoal",
    "Lavender",
    "Mushroom Brown",
    "Rose",
    "Turquoise Blue",
    "Sea Green",
    "Burgundy",
    "Yellow",
)

SEASON = ("Not Specified", "Spring", "Summer", "Fall", "Winter")
USAGE = (
    "Not Specified",
    "Formal",
    "Sports",
    "Travel",
    "Home",
    "Casual",
    "Ethnic",
    "Party",
    "Smart Casual",
)


def get_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img


def main():
    st.subheader("Instruction")
    st.markdown(
        "You can find the products you want by using the traditional category search."
    )

    c1, c2 = st.columns([1, 3])

    number_results = c1.number_input("Top-N-Search:", value=50, min_value=10, step=10)

    c1.subheader("Categories")
    gender = c1.selectbox("Gender:", options=GENDER, key="gender")
    masterCategory = c1.selectbox(
        "Product Main Category:", options=MASTERCATEGORY, key="mastercat"
    )
    subCategory = c1.selectbox(
        "Product Sub-Category:", options=SUBCATEGORY[masterCategory], key="subcat"
    )
    articleType = c1.selectbox(
        "Product Type:", options=ARTICLETTYPE[subCategory], key="type"
    )
    baseColour = c1.selectbox("Product Colour:", options=BASECOLOUR, key="colour")
    season = c1.selectbox("Product Season:", options=SEASON, key="season")
    year = c1.text_input("Product Year:", value="Not Specified", key="year")
    usage = c1.selectbox("Product Usage:", options=USAGE, key="usage")
    filter_button = c1.button("Filter")

    if filter_button:
        c2.subheader("Results")
        data_load_state = c2.empty()
        data_load_state.markdown("Searching results...")
        df = images_styles
        if gender != "Not Specified":
            df = df.where(lambda t: t["gender"] == gender)
        if masterCategory != "Not Specified":
            df = df.where(lambda t: t["mastercategory"] == masterCategory)
        if subCategory != "Not Specified":
            df = df.where(lambda t: t["subcategory"] == subCategory)
        if articleType != "Not Specified":
            df = df.where(lambda t: t["articletype"] == articleType)
        if baseColour != "Not Specified":
            df = df.where(lambda t: t["basecolour"] == baseColour)
        if season != "Not Specified":
            df = df.where(lambda t: t["season"] == season)
        if year != "Not Specified":
            df = df.where(lambda t: t["year"] == year)
        if usage != "Not Specified":
            df = df.where(lambda t: t["usage"] == usage)

        result = df[:number_results]
        data_load_state.markdown(
            f"**{len(list(result))} Products Found**: ... Printing images..."
        )
        captions = [row["productdisplayname"] for row in result]
        pool = Pool(1)
        images = pool.map(get_image_from_url, [row["link"] for row in result])
        c2.image(images, width=200, caption=captions)
        data_load_state.markdown(f"**{len(list(result))} Products Found**")


if __name__ == "__main__":
    main()
