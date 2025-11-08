async function predict() {
    let ticker = document.getElementById("ticker").value;

    if (!ticker) {
        alert("Enter a stock/crypto ticker!");
        return;
    }

    document.getElementById("resultImage").style.display = "none";

    const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker })
    });

    const data = await response.json();

    let img = document.getElementById("resultImage");
    img.src = "data:image/png;base64," + data.image;
    img.style.display = "block";
}
