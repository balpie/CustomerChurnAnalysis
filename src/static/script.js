document.addEventListener("DOMContentLoaded", function () {

    const internetService = document.querySelector(
        'select[name="InternetService"]'
    );

    const dependentServices = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ];

    function updateInternetDependentFields() {

        const noInternet = internetService.value === "No";

        dependentServices.forEach(function (serviceName) {

            const select = document.querySelector(
                `select[name="${serviceName}"]`
            );

            if (noInternet) {
                select.value = "No";
                select.parentNode.style.display = "none";
            } else {
                select.parentNode.style.display = "flex";
            }
        });
    }


    // Stato iniziale: campi bloccati finché non viene scelto InternetService
    if (internetService.value === "") {
        dependentServices.forEach(function (serviceName) {
            const select = document.querySelector(
                `select[name="${serviceName}"]`
            );
            select.readonly = true;
        });
    } else {
        updateInternetDependentFields();
    }


    // Aggiornamento quando cambia InternetService
    internetService.addEventListener(
        "change",
        updateInternetDependentFields
    );

});
