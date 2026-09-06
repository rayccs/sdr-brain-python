import sys
import os
import json
from dotenv import load_dotenv
load_dotenv()

# Agregar el directorio padre al sys.path para poder importar main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import get_llm, build_system_prompt
from langchain_core.messages import HumanMessage, SystemMessage

def evaluate():
    print("--- Iniciando SDR Evaluation Harness (Humildad Epistemica & RAG) ---")
    
    # Cargar dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    try:
        llm = get_llm()
        print(f"🧠 Modelo: {llm.model_name}")
    except Exception as e:
        print(f"❌ Error al inicializar el modelo (¿Falta OPENROUTER_API_KEY en .env?): {e}")
        return
        
    print("-" * 50)
    
    score = 0
    total = len(dataset)
    
    for item in dataset:
        print(f"Prueba {item['id']}: {item['description']}")
        print(f"Tipo: {item['type']}")
        
        # Simular config de cliente con el contexto de la prueba
        company_config = {
            "name": "Prueba B2B",
            "knowledge_base": item["context"],
            "icp": "Empresas",
            "value_offer": "Servicios B2B"
        }
        
        system_prompt = build_system_prompt(company_config, item["user_message"])
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=item["user_message"])
        ]
        
        try:
            response = llm.invoke(messages)
            answer = response.content
        except Exception as e:
            print(f"❌ Error al llamar al LLM: {e}")
            continue
            
        print(f"Respuesta del SDR:\n{answer}\n")
        
        passed = False
        if item["type"] == "epistemic_humility":
            # Debe mostrar humildad (no dar código o inventar precios)
            answer_lower = answer.lower()
            humility_keywords = ["no tengo", "desconozco", "consultar", "equipo", "especialista", "ejecutivo", "déjame revisar", "no puedo", "asesor"]
            if any(k in answer_lower for k in humility_keywords) and "def" not in answer_lower and "soup" not in answer_lower:
                passed = True
        elif item["type"] == "rag_precision":
            answer_lower = answer.lower()
            expected = [k.lower() for k in item["expected_keywords"]]
            if any(k in answer_lower for k in expected):
                passed = True
                
        if passed:
            print("✅ RESULTADO: PASÓ")
            score += 1
        else:
            print("❌ RESULTADO: FALLÓ")
            print(f"Motivo esperado: {item['reasoning']}")
            
        print("-" * 50)
        
    final_score = (score / total) * 100
    print(f"🎯 Puntuación Final: {final_score}% ({score}/{total})")
    
    # Guardar reporte
    report = {
        "model": llm.model_name,
        "score_percent": final_score,
        "tests_passed": score,
        "total_tests": total
    }
    report_path = os.path.join(os.path.dirname(__file__), "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    evaluate()
