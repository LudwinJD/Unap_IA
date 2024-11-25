import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class AdmissionAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.features = None
        self.feature_names = None
        
    def load_data(self, filepath: str) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_csv(filepath, sep=';', encoding='utf-8', low_memory=False)
            logging.info(f"Datos cargados exitosamente con dimensiones: {df.shape}")
            
            required_columns = ['FECHA_NACIMIENTO', 'PUNTAJE', 'SEXO', 'ESTADO_CIVIL', 
                              'TIPO_COLEGIO', 'MODALIDAD_INGRESO', 'ESCUELA_PROFESIONAL']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logging.error(f"Faltan columnas requeridas: {missing_columns}")
                return None
                
            return df
            
        except Exception as e:
            logging.error(f"Error al cargar datos: {str(e)}")
            return None

    def process_birth_date(self, df: pd.DataFrame) -> pd.Series:
        current_year = datetime.now().year
        
        try:
            # Primero intentamos limpiar el formato de fecha
            df['FECHA_NACIMIENTO'] = df['FECHA_NACIMIENTO'].astype(str).str.strip()
            df['FECHA_NACIMIENTO'] = df['FECHA_NACIMIENTO'].str.replace('/', '')
            
            # Intentamos convertir a datetime
            df['FECHA_NACIMIENTO'] = pd.to_datetime(df['FECHA_NACIMIENTO'], format='%Y%m%d', errors='coerce')
            ages = current_year - df['FECHA_NACIMIENTO'].dt.year
            
            # Filtrar edades no razonables
            ages = ages.where((ages >= 15) & (ages <= 70), np.nan)
            
            return ages
            
        except Exception as e:
            logging.warning(f"Error al procesar fechas de nacimiento: {str(e)}")
            # Alternativa: intentar extraer el año como string
            try:
                years = df['FECHA_NACIMIENTO'].astype(str).str[:4].astype(float)
                ages = current_year - years
                return ages.where((ages >= 15) & (ages <= 70), np.nan)
            except:
                logging.error("No se pudieron procesar las fechas de nacimiento")
                return pd.Series(np.nan, index=df.index)

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepara los datos para machine learning, incluyendo ingeniería de características y preprocesamiento.
        """
        logging.info("Iniciando preparación de datos...")
        df_ml = df.copy()
        
        # Calcular edad
        df_ml['EDAD'] = self.process_birth_date(df_ml)
        logging.info("Cálculo de edad completado")
        
        # Procesar puntajes
        df_ml['PUNTAJE'] = pd.to_numeric(df_ml['PUNTAJE'], errors='coerce')
        df_ml = df_ml.dropna(subset=['PUNTAJE'])
        logging.info(f"Después de eliminar puntajes NA, dimensiones: {df_ml.shape}")
        
        # Crear variable objetivo
        score_threshold = df_ml['PUNTAJE'].median()
        df_ml['EXITO_ADMISION'] = (df_ml['PUNTAJE'] > score_threshold).astype(int)
        
        # Definir características
        self.features = ['EDAD', 'SEXO', 'ESTADO_CIVIL', 'TIPO_COLEGIO', 
                        'MODALIDAD_INGRESO', 'ESCUELA_PROFESIONAL']
        
        # Separar características numéricas y categóricas
        numeric_features = ['EDAD']  # Solo EDAD es numérica
        categorical_features = ['SEXO', 'ESTADO_CIVIL', 'TIPO_COLEGIO', 
                              'MODALIDAD_INGRESO', 'ESCUELA_PROFESIONAL']
        
        logging.info(f"Características numéricas: {numeric_features}")
        logging.info(f"Características categóricas: {categorical_features}")
        
        # Manejar valores faltantes para características numéricas
        if numeric_features:
            numeric_imputer = SimpleImputer(strategy='mean')
            df_ml[numeric_features] = numeric_imputer.fit_transform(df_ml[numeric_features].values.reshape(-1, 1))
        
        # Manejar valores faltantes para características categóricas
        for feature in categorical_features:
            df_ml[feature] = df_ml[feature].fillna('DESCONOCIDO')
            le = LabelEncoder()
            df_ml[feature] = le.fit_transform(df_ml[feature].astype(str))
            self.label_encoders[feature] = le
        
        self.feature_names = self.features.copy()
        logging.info("Preparación de datos completada")
        
        return df_ml, self.features

    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Tuple[LogisticRegression, dict]:
        """
        Entrena el modelo de regresión logística con validación cruzada.
        """
        logging.info("Iniciando entrenamiento del modelo...")
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Escalar características
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Entrenar modelo
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.model.fit(X_train_scaled, y_train)
        
        # Realizar validación cruzada
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        # Calcular accuracy
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calcular curva ROC
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        results = {
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'cv_scores': cv_scores,
            'accuracy': accuracy,
            'roc_data': (fpr, tpr, roc_auc)
        }
        
        logging.info(f"Entrenamiento del modelo completado. Accuracy: {accuracy:.3f}")
        return self.model, results

    def evaluate_model(self, results: dict) -> None:
        """
        Evalúa el rendimiento del modelo con varias métricas y visualizaciones.
        """
        logging.info("Iniciando evaluación del modelo...")
        
        X_test_scaled = results['X_test_scaled']
        y_test = results['y_test']
        cv_scores = results['cv_scores']
        accuracy = results['accuracy']
        fpr, tpr, roc_auc = results['roc_data']
        
        # Imprimir resultados de validación cruzada y accuracy
        logging.info(f"\nPuntajes de validación cruzada: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        logging.info(f"Accuracy del modelo: {accuracy:.3f}")
        
        # Reporte de clasificación
        y_pred = self.model.predict(X_test_scaled)
        print("\nReporte de Clasificación:")
        print(classification_report(y_test, y_pred))
        
        # Graficar accuracy junto con curva ROC
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Curva ROC
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('Tasa de Falsos Positivos')
        ax1.set_ylabel('Tasa de Verdaderos Positivos')
        ax1.set_title('Curva ROC')
        ax1.legend(loc="lower right")
        
        # Accuracy plot
        ax2.bar(['Accuracy'], [accuracy], color='skyblue')
        ax2.set_ylim([0, 1])
        ax2.set_title('Accuracy del Modelo')
        ax2.text('Accuracy', accuracy/2, f'{accuracy:.3f}', ha='center', va='center')
        
        plt.tight_layout()
        plt.show()
        
        # Matriz de confusión
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusión')
        plt.ylabel('Etiqueta Verdadera')
        plt.xlabel('Etiqueta Predicha')
        plt.show()
        
        # Importancia de características
        self.plot_feature_importance()
        
        logging.info("Evaluación del modelo completada")

    def plot_feature_importance(self) -> None:
        """
        Grafica la importancia de las características basada en los coeficientes del modelo.
        """
        importance = pd.DataFrame({
            'característica': self.feature_names,
            'importancia': np.abs(self.model.coef_[0])
        }).sort_values('importancia', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance, x='importancia', y='característica')
        plt.title('Importancia de las Características')
        plt.xlabel('Valor Absoluto del Coeficiente')
        plt.tight_layout()
        plt.show()

    def analyze_fairness(self, df: pd.DataFrame, sensitive_features: List[str]) -> Dict:
        """
        Analiza la equidad del modelo a través de diferentes grupos demográficos.
        """
        fairness_metrics = {}
        
        for feature in sensitive_features:
            if feature not in df.columns:
                continue
                
            groups = df[feature].unique()
            group_metrics = {}
            
            for group in groups:
                mask = df[feature] == group
                group_data = df[mask]
                
                if len(group_data) < 50:  # Omitir grupos pequeños
                    continue
                    
                X_group = group_data[self.features]
                X_group_scaled = self.scaler.transform(X_group)
                y_group = group_data['EXITO_ADMISION']
                
                y_pred = self.model.predict(X_group_scaled)
                group_accuracy = accuracy_score(y_group, y_pred)
                
                metrics = {
                    'tamaño': len(group_data),
                    'tasa_éxito': y_group.mean(),
                    'tasa_predicha': y_pred.mean(),
                    'disparidad': y_pred.mean() - y_group.mean(),
                    'accuracy': group_accuracy
                }
                
                group_metrics[group] = metrics
            
            fairness_metrics[feature] = group_metrics
        
        return fairness_metrics

def main():
    """
    Función principal de ejecución para el pipeline de análisis de admisiones.
    """
    analyzer = AdmissionAnalyzer()
    
    # Cargar datos
    df = analyzer.load_data('Ingresantes_unap.csv')
    if df is None:
        return
    
    # Preparar datos
    df_ml, features = analyzer.prepare_data(df)
    if df_ml is None:
        return
    
    # Entrenar modelo
    X = df_ml[features]
    y = df_ml['EXITO_ADMISION']
    model, results = analyzer.train_model(X, y)
    
    # Evaluar modelo
    analyzer.evaluate_model(results)
    
    # Analizar equidad
    sensitive_features = ['TIPO_COLEGIO', 'MODALIDAD_INGRESO', 'ESTADO_CIVIL']
    fairness_metrics = analyzer.analyze_fairness(df_ml, sensitive_features)
    
    # Imprimir análisis de equidad
    print("\nAnálisis de Equidad:")
    for feature, metrics in fairness_metrics.items():
        print(f"\nAnálisis de {feature}:")
        for group, group_metrics in metrics.items():
            print(f"\nGrupo: {group}")
            print(f"Tamaño del grupo: {group_metrics['tamaño']}")
            print(f"Tasa de éxito real: {group_metrics['tasa_éxito']:.2%}")
            print(f"Tasa de éxito predicha: {group_metrics['tasa_predicha']:.2%}")
            print(f"Disparidad: {group_metrics['disparidad']:.2%}")
            print(f"Accuracy del grupo: {group_metrics['accuracy']:.2%}")

if __name__ == "__main__":
    main()