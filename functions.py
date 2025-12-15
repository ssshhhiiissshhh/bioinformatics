def dist_levenshtein(seq1, seq2):    
    # Инициализация матрицы
    m, n = len(seq1), len(seq2)
    score_matrix = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    
    # Заполнение первой строки и первого столбца
    for i in range(m + 1):
        score_matrix[i][0] = i
    for j in range(n + 1):
        score_matrix[0][j] = j
    
    # Заполнение матрицы
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Если символы совпадают, стоимость = 0, иначе 1 (замена)
            match = score_matrix[i-1][j-1] + (0 if seq1[i-1] == seq2[j-1] else 1)
            delete = score_matrix[i-1][j] + 1
            insert = score_matrix[i][j-1] + 1
            
            # Выбор минимального значения
            score_matrix[i][j] = min(match, delete, insert)
       
    return score_matrix[m][n]


def needleman_wunsch(seq1, seq2, match_score=1, mismatch=-1, gap=-1):    
    # Инициализация матрицы
    m, n = len(seq1), len(seq2)
    score_matrix = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    
    # Заполнение первой строки и первого столбца
    for i in range(m + 1):
        score_matrix[i][0] = i * gap
    for j in range(n + 1):
        score_matrix[0][j] = j * gap
    
    # Заполнение матрицы
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Вычисление трех возможных значений
            match = score_matrix[i-1][j-1] + (match_score if seq1[i-1] == seq2[j-1] else mismatch)
            delete = score_matrix[i-1][j] + gap
            insert = score_matrix[i][j-1] + gap
            
            # Выбор максимального значения
            score_matrix[i][j] = max(match, delete, insert)
    
    # Обратный ход
    aligned_seq1 = ""
    aligned_seq2 = ""
    alignment_line = ""
    
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and score_matrix[i][j] == score_matrix[i-1][j-1] + (match_score if seq1[i-1] == seq2[j-1] else mismatch):
            # Диагональный переход (совпадение или несовпадение)
            aligned_seq1 = seq1[i-1] + aligned_seq1
            aligned_seq2 = seq2[j-1] + aligned_seq2
            if seq1[i-1] == seq2[j-1]:
                alignment_line = "|" + alignment_line
            else:
                alignment_line = "*" + alignment_line
            i -= 1
            j -= 1
        elif i > 0 and score_matrix[i][j] == score_matrix[i-1][j] + gap:
            # Вертикальный переход (гэп во второй последовательности)
            aligned_seq1 = seq1[i-1] + aligned_seq1
            aligned_seq2 = "-" + aligned_seq2
            alignment_line = " " + alignment_line
            i -= 1
        else:
            # Горизонтальный переход (гэп в первой последовательности)
            aligned_seq1 = "-" + aligned_seq1
            aligned_seq2 = seq2[j-1] + aligned_seq2
            alignment_line = " " + alignment_line
            j -= 1
    
    return aligned_seq1, alignment_line, aligned_seq2, score_matrix[m][n]


def find_min_pair(D):
    n = len(D)
    min_val = np.inf
    min_pair = (0, 0)
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] < min_val:
                min_val = D[i, j]
                min_pair = (i, j)
    return min_pair


def WPGMA(matrix, labels):
    n = len(matrix)

    # Восстанавливаем матрицу на случай, если подавалась нижнетреугольная
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            D[i, j] = matrix[i][j]
            D[j, i] = matrix[i][j]

    # Высоты кластеров
    heights = {label: 0.0 for label in labels}
    nearest_clusters = []
    # Алгоритм (WPGMA)
    while len(D) > 1:
        # Находим два ближайших кластера
        i, j = find_min_pair(D)
        ci, cj = labels[i], labels[j]
        dist = D[i, j] / 2
        nearest_clusters.append((i, j))

        # Создаем новое имя в формате Newick
        new_label = f"({ci}:{dist - heights[ci]},{cj}:{dist - heights[cj]})"
        heights[new_label] = dist

        # Вычисляем новое расстояние до других кластеров
        new_row = (D[i, :] + D[j, :]) / 2
        new_row = np.delete(new_row, [i, j])

        # Обновляем матрицу расстояний
        D = np.delete(D, [i, j], axis=0)
        D = np.delete(D, [i, j], axis=1)
        D = np.vstack((D, new_row))
        new_col = np.append(new_row, [0.0])
        D = np.column_stack((D, new_col))

        # Обновляем метки
        labels = [lab for k, lab in enumerate(labels) if k not in (i, j)] + [new_label]

    return labels[0] + ";", nearest_clusters


def k2p_distance(seq1: str, seq2: str) -> float:
    """
    Вычисляет расстояние Кимуры 2-параметра между двумя последовательностями.
    
    Args:
        seq1: Первая последовательность
        seq2: Вторая последовательность
        
    Returns:
        Расстояние Кимуры 2-параметра
    """
    if len(seq1) != len(seq2):
        raise ValueError("Последовательности должны иметь одинаковую длину")
    
    # Определяем пурины и пиримидины
    purines = {'A', 'G'}
    pyrimidines = {'C', 'T', 'U'}  # U для РНК
    
    # Счетчики
    transitions = 0  # транзиции: A↔G, C↔T
    transversions = 0  # трансверсии: A/G ↔ C/T
    total = 0  # общее количество сравненных позиций
    
    for n1, n2 in zip(seq1, seq2):
        # Пропускаем гэпы и неопределенные нуклеотиды
        if n1 == '-' or n2 == '-' or n1 == 'N' or n2 == 'N':
            continue
        
        if n1 == n2:
            total += 1
            continue
        
        total += 1
        
        # Определяем тип замены
        if (n1 in purines and n2 in purines) or (n1 in pyrimidines and n2 in pyrimidines):
            # Транзиция (внутри одного типа)
            transitions += 1
        else:
            # Трансверсия (между типами)
            transversions += 1
    
    if total == 0:
        return 0.0
    
    # Вычисляем доли
    P = transitions / total
    Q = transversions / total
    
    # Формула Кимуры 2-параметра
    # d = -1/2 * ln(1 - 2P - Q) - 1/4 * ln(1 - 2Q)
    
    # Проверяем, чтобы аргументы логарифмов были положительными
    term1 = 1 - 2*P - Q
    term2 = 1 - 2*Q
    
    if term1 <= 0 or term2 <= 0:
        # Если аргументы неположительные, используем Jukes-Cantor как запасной вариант
        # или возвращаем большое расстояние
        diff_sites = transitions + transversions
        p = diff_sites / total if total > 0 else 0
        if p >= 0.75:
            return float('inf')
        return -0.75 * math.log(1 - (4/3) * p)
    
    d = -0.5 * math.log(term1) - 0.25 * math.log(term2)
    
    return d


def parse_sequence_file(filename):
    organisms = []
    amino_acids = []
    nucleotides = []
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Если строка начинается с '>', это название организма
                if line.startswith('>'):
                    organism_name = line[1:].strip()  # Убираем '>' и лишние пробелы
                    organisms.append(organism_name)
                    
                    # Следующая строка - аминокислотная последовательность
                    if i + 1 < len(lines):
                        aa_seq = lines[i + 1].strip()
                        amino_acids.append(aa_seq)
                    
                    # Через одну строку - нуклеотидная последовательность
                    if i + 2 < len(lines):
                        nt_seq = lines[i + 2].strip()
                        nucleotides.append(nt_seq)
                    
                    i += 3  # Переходим к следующему блоку
                else:
                    i += 1  # Пропускаем строки без '>'
                    
    except FileNotFoundError:
        print(f"Ошибка: Файл '{filename}' не найден")
        return [], [], []
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return [], [], []
    
    return organisms, amino_acids, nucleotides


def levenshtein_matrix(sequences):
    n = len(sequences)
    D = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = dist_levenshtein(sequences[i], sequences[j])
            D[i, j] = dist
            D[j, i] = dist
    
    return D


def remove_gapped_columns(aligned_sequences):
    """
    Удаляет все колонки, где есть хотя бы один '-'
    Возвращает очищенные последовательности (без гэпов вообще)
    """
    if not aligned_sequences:
        return []
    
    length = len(aligned_sequences[0])
    n = len(aligned_sequences)
    
    cleaned = [""] * n
    
    for col in range(length):
        column = [aligned_sequences[i][col] for i in range(n)]
        if '-' not in column:  # только если ни одного гэпа
            for i in range(n):
                cleaned[i] += column[i]
    
    return cleaned


def k2p_matrix(sequences):
    n = len(sequences)
    D = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = k2p_distance(sequences[i], sequences[j])
            D[i, j] = dist
            D[j, i] = dist
    
    return D