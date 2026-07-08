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
                select.value = "No internet service";
                select.disabled = true;
            } else {
                select.disabled = false;

                // se era stato impostato automaticamente prima
                if (select.value === "No internet service") {
                    select.value = "No";
                }
            }
        });
    }


    // Stato iniziale: campi bloccati finché non viene scelto InternetService
    if (internetService.value === "") {
        dependentServices.forEach(function (serviceName) {
            const select = document.querySelector(
                `select[name="${serviceName}"]`
            );
            select.disabled = true;
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
