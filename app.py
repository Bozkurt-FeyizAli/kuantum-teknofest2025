# Gerekli kütüphaneyi içe aktarın
from qiskit_ibm_provider import IBMProvider

# Kopyaladığınız API anahtarını tırnak işaretlerinin arasına yapıştırın
my_token = 'ApiKey-4bb3db24-5d44-4e4f-b74e-1d849cb95980'

# save_account fonksiyonunu kullanarak anahtarınızı bilgisayarınıza kaydedin.
# Bu işlemi SADECE BİR KEZ yapmanız yeterlidir.
# overwrite=True parametresi, eğer daha önceden kayıtlı bir anahtar varsa üzerine yazar.
IBMProvider.save_account(token=my_token, overwrite=True)

print("API anahtarınız başarıyla bilgisayarınıza kaydedildi.")
print("Artık bu kodu tekrar çalıştırmanıza gerek yok.")