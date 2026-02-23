function submitForm() {
    const landArea = document.getElementById('landArea').value;
    const location = document.getElementById('location').value;
    const cropType = document.getElementById('cropType').value;
    const soilPH = document.getElementById('soilPH').value;

    if (!landArea || !location || !cropType || !soilPH) {
        alert("Please fill in all fields.");
        return;
    }

    alert(`Form Submitted!\nLand Area: ${landArea}\nLocation: ${location}\nCrop Type: ${cropType}\nSoil pH: ${soilPH}`);
    // Here you would typically send this data to the backend
    console.log("Submitting data:", { landArea, location, cropType, soilPH });
}
