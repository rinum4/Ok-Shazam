# -*- coding: utf-8 -*-

import hashlib
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from termcolor import colored
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import (generate_binary_structure, iterate_structure, binary_erosion)
from operator import itemgetter

IDX_FREQ_I = 0
IDX_TIME_J = 1

# Частота дискретизации, связанная с условиями Найквиста, которая влияет на
# диапазон частот, которые мы можем обнаружить.
DEFAULT_FS = 44100

# Размер окна БПФ, влияет на детализацию частоты
DEFAULT_WINDOW_SIZE = 4096

# Соотношение, при котором каждое последовательное окно перекрывает последнее и второе.
# следующее окно. Более высокое перекрытие позволит повысить детализацию смещения
# и совпадение, но потенциально приводит к большему количеству хэшей
DEFAULT_OVERLAP_RATIO = 0.5

# Степень, в которой хэш может быть сопряжен со своими соседями -
# более высокое значение приводит к увеличению хэшей, но потенциально более высокую точность.
DEFAULT_FAN_VALUE = 15

# Минимальная амплитуда в спектрограмме для того, чтобы считаться пиком.
# Может быть повышено, чтобы уменьшить количество хэшей, но может отрицательно сказаться
# влияет на точность.
DEFAULT_AMP_MIN = 10

# Количество ячеек вокруг амплитудного пика в спектрограмме по порядку,
# чтобы считать его спектральным пиком. Более высокие значения означают меньше
# хэшей и более быстрое сопоставление, но потенциально могут повлиять на точность.
PEAK_NEIGHBORHOOD_SIZE = 20

# Пороговые значения того, насколько близко или далеко могут быть пики во времени по порядку
# быть спаренным в качестве одного хэша. Если ваш Максимум слишком низок, то более высокие значения
# DEFAULT_FAN_VALUE могут работать не так, как ожидалось.
MIN_HASH_TIME_DELTA = 0
MAX_HASH_TIME_DELTA = 200

# Если Истина, то пики будут сортироваться по времени для генерации хэшей;
# отсутствие сортировки сократит количество хэшей, но потенциально
# влияет на производительность.
PEAK_SORT = True

# Количество битов, которые нужно выбросить из передней части хэша SHA1 в
# расчет хэшей. Чем больше вы выбрасываете, тем меньше места для хранения, но
# потенциально это более высокое количество ошибок и неверные классификации при идентификации песен.
FINGERPRINT_REDUCTION = 20

def fingerprint(channel_samples, Fs=DEFAULT_FS,
                wsize=DEFAULT_WINDOW_SIZE,
                wratio=DEFAULT_OVERLAP_RATIO,
                fan_value=DEFAULT_FAN_VALUE,
                amp_min=DEFAULT_AMP_MIN,
                plots=False):

    # выводим график
    if plots:
      plt.plot(channel_samples)
      plt.title('%d samples' % len(channel_samples))
      plt.xlabel('time (s)')
      plt.ylabel('amplitude (A)')
      plt.show()
      plt.gca().invert_yaxis()

    # БПФ канала, логарифмическое преобразования сигнала,
    # нахождение локальных максимумов, и затем возвращаются
    # локально чувствительные хэши.

    # БПФ сигнала и извлечения частотных составляющих
    # строим график углового спектра сегментов внутри сигнала в виде цветовой карты
    arr2D = mlab.specgram(
        channel_samples,
        NFFT=wsize,
        Fs=Fs,
        window=mlab.window_hanning,
        noverlap=int(wsize * wratio))[0]

    # выводим график
    if plots:
      plt.plot(arr2D)
      plt.title('FFT')
      plt.show()

    # применяем логарифмическое преобразования сигнала т.к. specgram() возвращает линейный массив
    arr2D = 10 * np.log10(arr2D) #
    arr2D[arr2D == -np.inf] = 0  # бесконечность заменим на 0

    # ищем локальные максимумы
    local_maxima = get_2D_peaks(arr2D, plot=plots, amp_min=amp_min)

    msg = '   local_maxima: %d of frequency & time pairs'
    print(colored(msg, attrs=['dark']) % len(local_maxima))

    # возвращаем хэш
    return generate_hashes(local_maxima, fan_value=fan_value)

def get_2D_peaks(arr2D, plot=False, amp_min=DEFAULT_AMP_MIN):
    # http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.morphology.iterate_structure.html#scipy.ndimage.morphology.iterate_structure
    struct = generate_binary_structure(2, 1)
    neighborhood = iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)

    # найдите локальные максимумы, используя нашу форму фильтра
    local_max = maximum_filter(arr2D, footprint = neighborhood) == arr2D
    background = (arr2D == 0)
    # бинарная эррозия матрицы
    eroded_background = binary_erosion(background, structure = neighborhood,
                                       border_value=1)

    # Булева маска arr2D с Истиной на пиках
    detected_peaks = local_max ^ eroded_background

    # вычленяем пики
    amps = arr2D[detected_peaks]
    j, i = np.where(detected_peaks)

    # фильтруем пики
    amps = amps.flatten()
    peaks = zip(i, j, amps)
    peaks_filtered = [x for x in peaks if x[2] > amp_min]  # частота, время, апмлитуда

    # получить индексы для частоты и времени
    frequency_idx = [x[1] for x in peaks_filtered]
    time_idx = [x[0] for x in peaks_filtered]

    # график пиков
    if plot:
      fig, ax = plt.subplots()
      ax.imshow(arr2D)
      ax.scatter(time_idx, frequency_idx)
      ax.set_xlabel('Time')
      ax.set_ylabel('Frequency')
      ax.set_title("Spectrogram")
      plt.gca().invert_yaxis()
      plt.show()

    return zip(frequency_idx, time_idx)

# Структура из списка хэшей: [sha1_hash[0:20] временное_смещение]
# например: [(e05b341a9b77a51fd26, 32), ... ]
def generate_hashes(peaks, fan_value=DEFAULT_FAN_VALUE):
    if PEAK_SORT:
      peaks.sort(key=itemgetter(1))

    # для каждого пика
    for i in range(len(peaks)):
      # для каждого соседа
      for j in range(1, fan_value):
        if (i + j) < len(peaks):

          # получим текущее и следующее пиковое значение частоты
          freq1 = peaks[i][IDX_FREQ_I]
          freq2 = peaks[i + j][IDX_FREQ_I]

          # получим текущее и следующее смещение времени пика
          t1 = peaks[i][IDX_TIME_J]
          t2 = peaks[i + j][IDX_TIME_J]

          # получим разницу по времени
          t_delta = t2 - t1

          # если разница между минимум и максимом считаем хэш
          if t_delta >= MIN_HASH_TIME_DELTA and t_delta <= MAX_HASH_TIME_DELTA:
            h = hashlib.sha1("%s|%s|%s" % (str(freq1), str(freq2), str(t_delta)))
            # обрезаем хэш
            yield (h.hexdigest()[0:FINGERPRINT_REDUCTION], t1)
