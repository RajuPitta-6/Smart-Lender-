const API_URL = "https://smart-lender-g9fm.onrender.com/";

const form = document.getElementById("loanForm");
const result = document.getElementById("result");

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    result.style.display = "block";
    result.className = "";
    result.innerHTML = "<h2>Predicting...</h2>";

    const applicant = {
        applicant_name: document.getElementById("applicant_name").value,
        gender: document.getElementById("gender").value,
        education: document.getElementById("education").value,
        self_employed: document.getElementById("self_employed").value,
        no_of_dependents: Number(document.getElementById("no_of_dependents").value),
        income_annum: Number(document.getElementById("income_annum").value),
        loan_amount: Number(document.getElementById("loan_amount").value),
        loan_term: Number(document.getElementById("loan_term").value),
        cibil_score: Number(document.getElementById("cibil_score").value)
    };

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(applicant)
        });

        if (!response.ok) {
            throw new Error("Prediction failed.");
        }

        const data = await response.json();

        result.className =
            data.prediction === "Approved"
                ? "approved"
                : "rejected";

        result.innerHTML = `
            <h2>${data.prediction}</h2>
            <p><strong>Confidence:</strong> ${data.confidence}%</p>
            <p><strong>Applicant:</strong> ${data.applicant_name}</p>
        `;

    } catch (error) {

        result.className = "rejected";

        result.innerHTML = `
            <h2>Error</h2>
            <p>${error.message}</p>
        `;
    }
});