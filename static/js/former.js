document.getElementById('addFormButton').addEventListener('click', function() {

    if (document.getElementById('formContainer').children.length >= 4) {
        alert("You can't have more than 4 forms!");
        return;
    } else {

    var formContainer = document.getElementById('formContainer');

    var cont = document.createElement('div');

    cont.setAttribute("class", "input-group");
    cont.setAttribute("style", "max-width: 50%");

    var prsi = document.createElement('input');
    prsi.setAttribute('name', 'PRSI' + (formContainer.children.length + 1));
    prsi.setAttribute('type', 'text');
    prsi.setAttribute('aria-label', 'RSI period');
    prsi.setAttribute('class', 'form-control');
    prsi.setAttribute('placeholder', 'RSI period');

    var pdrsi = document.createElement('input');
    pdrsi.setAttribute('name', 'PDRSI' + (formContainer.children.length + 1));
    pdrsi.setAttribute('type', 'text');
    pdrsi.setAttribute('aria-label', 'dRSI period');
    pdrsi.setAttribute('class', 'form-control');
    pdrsi.setAttribute('placeholder', 'dRSI period');

    cont.appendChild(prsi);
    cont.appendChild(pdrsi);

    formContainer.appendChild(cont);

    }
});