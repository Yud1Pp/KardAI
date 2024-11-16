// Fungsi untuk mengontrol tampilan dropdown buah
function toggleFruitQuantity() {
    var fruitSelect = document.getElementById('fruit');
    var fruitQuantityDiv = document.getElementById('fruitQuantity');
    if (fruitSelect.value === 'day' || fruitSelect.value === 'week' || fruitSelect.value === 'month') {
      fruitQuantityDiv.style.display = 'block';
    } else {
      fruitQuantityDiv.style.display = 'none';
    }
  }
  
  // Fungsi untuk mengontrol tampilan dropdown sayur
  function toggleVegetablesQuantity() {
    var vegetablesSelect = document.getElementById('vegetables');
    var vegetablesQuantityDiv = document.getElementById('vegetablesQuantity');
    if (vegetablesSelect.value === 'day' || vegetablesSelect.value === 'week' || vegetablesSelect.value === 'month') {
      vegetablesQuantityDiv.style.display = 'block';
    } else {
      vegetablesQuantityDiv.style.display = 'none';
    }
  }
  
  // Fungsi untuk mengontrol tampilan dropdown kentang goreng
  function toggleFriedQuantity() {
    var friedSelect = document.getElementById('fried');
    var friedQuantityDiv = document.getElementById('friedQuantity');
    if (friedSelect.value === 'day' || friedSelect.value === 'week' || friedSelect.value === 'month') {
      friedQuantityDiv.style.display = 'block';
    } else {
      friedQuantityDiv.style.display = 'none';
    }
  }
  
  // Fungsi untuk memperbarui nilai slider alkohol
  document.getElementById('slider').addEventListener('input', function (event) {
    document.getElementById('sliderValue').innerText = event.target.value;
  });
  