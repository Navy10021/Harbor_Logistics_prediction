import torch
import torch.nn as nn
import math

# Time2Vec: 시간의 선형 및 주기적 패턴을 학습하는 모듈
class Time2Vec(nn.Module):
    def __init__(self, in_features=1, out_features=64):
        """
        in_features: 입력 차원 (보통 시간 스탬프 1개)
        out_features: 출력 임베딩 차원. 첫 번째 차원은 선형항, 나머지는 주기항.
        """
        super(Time2Vec, self).__init__()
        self.out_features = out_features
        self.linear = nn.Linear(in_features, 1)  # 선형 항
        self.periodic = nn.Linear(in_features, out_features - 1)  # 주기적 항

    def forward(self, t):
        # t: (batch_size, seq_len, 1)
        linear_term = self.linear(t)  # 선형 항 출력: (batch, seq_len, 1)
        periodic_term = torch.sin(self.periodic(t))  # 주기 항 출력: (batch, seq_len, out_features-1)
        return torch.cat([linear_term, periodic_term], dim=-1)  # (batch, seq_len, out_features)

# Frequency-based Encoding: Fourier 변환 기반의 주파수 성분 인코딩
class FrequencyEncoding(nn.Module):
    def __init__(self, num_frequencies=4):
        """
        num_frequencies: 사용할 주파수 개수.
        각 주파수에 대해 sine와 cosine 항을 계산하여 총 2*num_frequencies 차원의 인코딩을 생성.
        """
        super(FrequencyEncoding, self).__init__()
        self.num_frequencies = num_frequencies
        # 고정 주파수 값: 1부터 num_frequencies까지 선형으로 증가
        frequencies = torch.linspace(1, num_frequencies, steps=num_frequencies)
        self.register_buffer('frequencies', frequencies)

    def forward(self, t):
        # t: (batch, seq_len, 1)
        freq = self.frequencies.view(1, 1, -1)  # (1, 1, num_frequencies)
        sin_component = torch.sin(2 * math.pi * freq * t)
        cos_component = torch.cos(2 * math.pi * freq * t)
        # 결과: (batch, seq_len, 2*num_frequencies)
        return torch.cat([sin_component, cos_component], dim=-1)

# Combined Positional Encoding: Time2Vec와 Frequency Encoding 결합
class CombinedPositionalEncoding(nn.Module):
    def __init__(self, time2vec_dim=64, num_frequencies=4):
        super(CombinedPositionalEncoding, self).__init__()
        self.time2vec = Time2Vec(in_features=1, out_features=time2vec_dim)
        self.freq_encoding = FrequencyEncoding(num_frequencies=num_frequencies)

    def forward(self, t):
        # t: (batch, seq_len, 1)
        time2vec_feat = self.time2vec(t)        # (batch, seq_len, time2vec_dim)
        freq_feat = self.freq_encoding(t)         # (batch, seq_len, 2*num_frequencies)
        # 결합: (batch, seq_len, time2vec_dim + 2*num_frequencies)
        return torch.cat([time2vec_feat, freq_feat], dim=-1)

# 트랜스포머 모델에 결합된 포지셔널 인코딩 적용 예제
class TransformerWithCustomPositionalEncoding(nn.Module):
    def __init__(self, input_dim, seq_len, d_model=128, nhead=8, num_layers=3,
                 time2vec_dim=64, num_frequencies=4):
        """
        input_dim: 입력 데이터의 feature 차원
        seq_len: 시퀀스 길이
        d_model: 트랜스포머 임베딩 차원
        nhead: 멀티-헤드 어텐션 헤드 수
        num_layers: 트랜스포머 인코더 레이어 수
        """
        super(TransformerWithCustomPositionalEncoding, self).__init__()
        # 결합된 포지셔널 인코딩 모듈
        self.pos_encoding = CombinedPositionalEncoding(time2vec_dim, num_frequencies)
        pos_dim = time2vec_dim + 2 * num_frequencies
        # 포지셔널 인코딩 차원을 d_model 차원으로 맞추기 위한 선형 변환
        self.pos_proj = nn.Linear(pos_dim, d_model)
        # 입력 임베딩
        self.input_projection = nn.Linear(input_dim, d_model)
        # 트랜스포머 인코더
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 최종 출력 투영 (예: 회귀나 시계열 예측)
        self.output_projection = nn.Linear(d_model, 1)

    def forward(self, x, t):
        """
        x: (batch, seq_len, input_dim) - 원시 입력 데이터
        t: (batch, seq_len, 1) - 시간 정보 (예: 정규화된 타임스탬프)
        """
        # 입력 임베딩
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        # 포지셔널 인코딩 계산 후 d_model 차원으로 프로젝션
        pos = self.pos_encoding(t)      # (batch, seq_len, pos_dim)
        pos = self.pos_proj(pos)          # (batch, seq_len, d_model)
        # 입력 임베딩과 포지셔널 인코딩 합산
        x = x + pos
        # 트랜스포머는 (seq_len, batch, d_model) 형태의 입력을 기대하므로 transpose
        x = x.transpose(0, 1)
        # 트랜스포머 인코더 통과
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)  # 다시 (batch, seq_len, d_model)로 변환
        # 최종 출력 계산
        output = self.output_projection(x)
        return output

# 사용 예시
if __name__ == "__main__":
    batch_size = 16
    seq_len = 50
    input_dim = 10

    # 임의의 입력 데이터와 시간 스탬프 (여기서는 0~1 사이 정규화된 값)
    x = torch.randn(batch_size, seq_len, input_dim)
    t = torch.linspace(0, 1, steps=seq_len).unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1)

    model = TransformerWithCustomPositionalEncoding(input_dim=input_dim, seq_len=seq_len)
    output = model(x, t)
    print("Output shape:", output.shape)
